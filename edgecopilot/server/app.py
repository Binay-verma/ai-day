"""Operator dashboard server (FastAPI + WebSocket).

Run:  python -m edgecopilot.server.app --source demo/demo_campus.mp4 --zone "Corridor B"

* ``GET  /``                         dashboard
* ``WS   /ws``                       live frames, hazard gauges, timings, new alerts
* ``GET  /api/alerts``               recent alerts (summaries)
* ``GET  /api/alerts/{id}``          full alert incl. snapshot + evidence images
* ``POST /api/alerts/{id}/feedback`` {"confirmed": bool, "note": str}  -> adapts zone bias
* ``POST /api/alerts/{id}/ask``      {"question": str}  -> grounded copilot answer
* ``POST /api/sensors``              IoT reading (ESP32 node posts here)
* ``GET  /api/status``               runtime stats, sensor health, config summary
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..copilot.events import EventLog
from ..copilot.qa import answer
from ..io.sensors import SensorHub
from ..io.video import VideoSource
from ..pipeline import Copilot, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class Feedback(BaseModel):
    confirmed: bool
    note: str = ""


class Question(BaseModel):
    question: str


class Runner:
    """Background thread: video -> copilot -> latest frame + broadcast queue."""

    def __init__(self, copilot: Copilot, source: str, loop: bool = True, realtime: bool = True, stride: int = 1,
                 sensor_csv: Optional[str] = None):
        self.copilot = copilot
        self.source = source
        self.loop = loop
        self.realtime = realtime
        self.stride = stride
        self.sensor_csv = sensor_csv
        self.latest_jpg: bytes = b""
        self.state: dict = {}
        self.fps_hist = deque(maxlen=30)
        self.lat_hist = deque(maxlen=300)
        self.subscribers: set = set()
        self.loop_ref: Optional[asyncio.AbstractEventLoop] = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _publish(self, msg: dict) -> None:
        if self.loop_ref is None:
            return
        for q in list(self.subscribers):
            try:
                self.loop_ref.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:
                pass

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                src = VideoSource(self.source, stride=self.stride, loop=False, realtime=self.realtime)
            except IOError as e:
                self.state = {"error": str(e)}
                time.sleep(2)
                continue
            self.copilot.reset()
            if self.sensor_csv:
                self.copilot.sensors.load_csv(self.sensor_csv)
            last = time.perf_counter()
            for frame, t in src:
                if self.stop.is_set():
                    return
                self.copilot.sensors.advance(t)
                res = self.copilot.process(frame, t)
                img = self.copilot.render(frame if frame.shape[1] <= self.copilot.max_width else
                                          cv2.resize(frame, (self.copilot.max_width, int(frame.shape[0] * self.copilot.max_width / frame.shape[1]))),
                                          res.people, res.hazards, res.assessments)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok:
                    self.latest_jpg = buf.tobytes()
                now = time.perf_counter()
                self.fps_hist.append(1.0 / max(now - last, 1e-6))
                last = now
                self.lat_hist.append(res.timings["total"])
                gauges = {}
                for a in res.assessments:
                    g = gauges.get(a.hazard)
                    if g is None or a.prob > g["prob"]:
                        gauges[a.hazard] = {"prob": round(a.prob, 3), "subject": a.subject, "active": a.active,
                                            "threshold": a.threshold,
                                            "top": [{"name": c.name, "c": round(c.contribution, 2), "v": round(c.value, 3)}
                                                    for c in a.top_cues(3, positive_only=False)]}
                self.state = {
                    "t": round(t, 2), "people": len(res.people), "motion": round(res.motion, 4),
                    "ran_detector": res.ran_detector, "timings": {k: round(v, 1) for k, v in res.timings.items()},
                    "fps": round(sum(self.fps_hist) / len(self.fps_hist), 1),
                    "latency_p50": round(sorted(self.lat_hist)[len(self.lat_hist) // 2], 1),
                    "gauges": gauges, "sensors": self.copilot.sensors.status(t),
                }
                self._publish({"type": "frame", "jpg": base64.b64encode(self.latest_jpg).decode(), "state": self.state})
                for al in res.new_alerts:
                    self._publish({"type": "alert", "alert": al.summary()})
            if not self.loop:
                break


def create_app(copilot: Copilot, runner: Optional[Runner]) -> FastAPI:
    app = FastAPI(title="Explainable Edge AI Copilot")

    @app.on_event("startup")
    async def _startup():
        if runner is not None:
            runner.loop_ref = asyncio.get_running_loop()
            runner.start()

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/api/status")
    def status():
        return {"state": runner.state if runner else {}, "zone": copilot.zone, "privacy": copilot.privacy,
                "models": {"pose": bool(copilot.pose), "fire_smoke": bool(copilot.fire)},
                "bias_offsets": {hz: m.bias_offset for hz, m in copilot.models.items()}}

    @app.get("/api/alerts")
    def alerts(n: int = 50):
        return [a.summary() for a in copilot.events.recent(n)]

    @app.get("/api/alerts/{alert_id}")
    def alert(alert_id: str):
        a = copilot.events.alerts.get(alert_id)
        if a is None:
            raise HTTPException(404, "unknown alert")
        d = a.summary()
        d["snapshot_jpg_b64"] = a.snapshot_jpg_b64
        d["evidence_jpg_b64"] = a.evidence_jpg_b64
        return d

    @app.post("/api/alerts/{alert_id}/feedback")
    def feedback(alert_id: str, fb: Feedback):
        r = copilot.feedback(alert_id, fb.confirmed, fb.note)
        if r is None:
            raise HTTPException(404, "unknown alert")
        return r

    @app.post("/api/alerts/{alert_id}/ask")
    def ask(alert_id: str, q: Question):
        a = copilot.events.alerts.get(alert_id)
        if a is None:
            raise HTTPException(404, "unknown alert")
        return answer(a.summary(), q.question)

    @app.post("/api/sensors")
    def sensors(msg: dict):
        if "t" not in msg:
            msg["t"] = runner.state.get("t", 0.0) if runner else 0.0
        copilot.sensors.push(msg)
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        if runner:
            runner.subscribers.add(q)
        try:
            await sock.send_text(json.dumps({"type": "hello", "alerts": [a.summary() for a in copilot.events.recent(20)]}, default=float))
            while True:
                msg = await q.get()
                if msg["type"] == "frame" and q.qsize() > 2:
                    continue  # drop stale frames for slow clients
                await sock.send_text(json.dumps(msg, default=float))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if runner:
                runner.subscribers.discard(q)

    return app


def main():
    ap = argparse.ArgumentParser(description="Explainable Edge AI Copilot dashboard")
    ap.add_argument("--source", default=os.path.join(ROOT, "demo", "demo_campus.mp4"), help="video file, webcam index or RTSP URL")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "campus.yaml"))
    ap.add_argument("--zone", default="Corridor B")
    ap.add_argument("--sensors", default=None, help="optional sensor CSV to replay in sync with the video")
    ap.add_argument("--privacy", choices=["raw", "blur", "skeleton"], default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-loop", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.privacy:
        cfg["runtime"]["privacy"] = args.privacy
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    cop = Copilot(cfg, root=ROOT, zone=args.zone, sensors=SensorHub(),
                  event_log=EventLog(os.path.join(ROOT, "logs", "events.jsonl")))
    sensor_csv = args.sensors
    if sensor_csv is None:
        cand = os.path.splitext(args.source)[0] + "_sensors.csv"
        sensor_csv = cand if os.path.exists(cand) else None
    runner = Runner(cop, args.source, loop=not args.no_loop, sensor_csv=sensor_csv)
    import uvicorn
    uvicorn.run(create_app(cop, runner), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
