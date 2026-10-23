"""Multi-camera runtime: one copilot per camera, self-healing streams, camera
health / tamper monitoring and pre/post-event clip recording.

Each :class:`CameraWorker` owns a video source, a :class:`Copilot` (its own
tracker, baselines and calibration for that zone) and a thread.  Production
concerns handled here:

* **Reconnect with exponential back-off** (1 s -> 30 s) when an RTSP/HTTP
  stream drops; files can loop for demos.
* **Health state machine**: ``connecting -> live -> stale -> offline`` from the
  age of the last frame, plus **tamper checks** - frozen image, blackout
  (covered lens / power loss to the IR lamp), loss of focus/sharpness (lens
  sprayed or rotated), each debounced and raised once as a ``camera_fault``.
* **Clip buffer**: a ring of privacy-filtered, down-scaled JPEG frames; when an
  alert fires the pre-event seconds plus the post-event seconds are written
  to an MP4 and handed to the encrypted store.
"""
from __future__ import annotations

import collections
import logging
import re
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .privacy import anonymize

log = logging.getLogger("edgecopilot.cameras")
_CRED = re.compile(r"(?i)([a-z][a-z0-9+.\-]*://)[^/@\s]*@")


def redact(text: str) -> str:
    """Remove user:password@ from any URL in a message before it is logged, stored or sent."""
    return _CRED.sub(r"\1***@", str(text))


@dataclass
class Health:
    state: str = "connecting"  # connecting | live | stale | offline | stopped
    fps: float = 0.0
    last_frame_wall: float = 0.0
    reconnects: int = 0
    frames: int = 0
    dropped: int = 0  # frames skipped because processing was slower than real time
    errors: int = 0   # frames whose processing raised (logged, thread keeps running)
    faults: Dict[str, bool] = field(default_factory=dict)  # frozen / blackout / defocus
    last_error: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["last_frame_age_s"] = round(time.time() - self.last_frame_wall, 1) if self.last_frame_wall else None
        return d


class TamperDetector:
    """Debounced image-health checks on a tiny grey thumbnail (< 0.5 ms/frame)."""

    def __init__(self, frozen_s: float = 15.0, blackout_s: float = 5.0, defocus_s: float = 10.0,
                 dark_level: float = 12.0, sharp_drop: float = 0.2):
        self.frozen_s, self.blackout_s, self.defocus_s = frozen_s, blackout_s, defocus_s
        self.dark_level, self.sharp_drop = dark_level, sharp_drop
        self.prev: Optional[np.ndarray] = None
        self.since: Dict[str, Optional[float]] = {"frozen": None, "blackout": None, "defocus": None}
        self.sharp_base: Optional[float] = None
        self.n = 0

    def __call__(self, frame: np.ndarray, t: float) -> Dict[str, bool]:
        g = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) \
            if frame.ndim == 3 else cv2.resize(frame, (64, 36))
        conds = {}
        conds["frozen"] = self.prev is not None and float(np.abs(g.astype(np.int16) - self.prev).mean()) < 0.05
        conds["blackout"] = float(g.mean()) < self.dark_level
        sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
        self.n += 1
        if self.sharp_base is None:
            self.sharp_base = sharp
        elif not conds["blackout"] and self.n > 50:
            # slow baseline so that a sudden loss of detail stands out
            self.sharp_base = 0.995 * self.sharp_base + 0.005 * sharp
        conds["defocus"] = self.n > 50 and self.sharp_base > 5 and sharp < self.sharp_drop * self.sharp_base
        self.prev = g.astype(np.int16)
        hold = {"frozen": self.frozen_s, "blackout": self.blackout_s, "defocus": self.defocus_s}
        out = {}
        for k, c in conds.items():
            if c:
                start = self.since[k]
                if start is None:
                    start = self.since[k] = t
                out[k] = (t - start) >= hold[k]
            else:
                self.since[k] = None
                out[k] = False
        return out


class ClipBuffer:
    """Ring buffer of JPEG frames; produces pre+post event clips."""

    def __init__(self, pre_s: float = 5.0, post_s: float = 10.0, width: int = 640, quality: int = 70):
        self.pre_s, self.post_s, self.width, self.quality = pre_s, post_s, width, quality
        self.frames: Deque[Tuple[float, bytes]] = collections.deque()
        self.pending: List[dict] = []  # clips waiting for post-event frames

    def add(self, frame: np.ndarray, t: float) -> List[Tuple[str, List[Tuple[float, bytes]]]]:
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            self.frames.append((t, buf.tobytes()))
        while self.frames and t - self.frames[0][0] > self.pre_s + self.post_s + 1:
            self.frames.popleft()
        done = []
        for p in list(self.pending):
            if t >= p["t_end"]:
                done.append((p["id"], [f for f in self.frames if p["t_start"] <= f[0] <= p["t_end"]]))
                self.pending.remove(p)
        return done

    def mark(self, alert_id: str, t: float) -> None:
        self.pending.append({"id": alert_id, "t_start": t - self.pre_s, "t_end": t + self.post_s})

    @staticmethod
    def encode_mp4(frames: List[Tuple[float, bytes]]) -> Optional[bytes]:
        if len(frames) < 2:
            return None
        fps = max(1.0, (len(frames) - 1) / max(frames[-1][0] - frames[0][0], 1e-3))
        first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
            for _, b in frames:
                img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and img.shape[:2] == first.shape[:2]:
                    vw.write(img)
            vw.release()
            with open(path, "rb") as f:
                return f.read()
        finally:
            os.remove(path)


class CameraWorker:
    def __init__(self, cam_id: str, source: Union[str, int], copilot, stride: int = 1, realtime: bool = True,
                 loop: bool = False, on_alert: Optional[Callable] = None, on_fault: Optional[Callable] = None,
                 on_clip: Optional[Callable] = None, on_frame: Optional[Callable] = None,
                 clip: Optional[ClipBuffer] = None, stale_s: float = 5.0, sensors_csv: Optional[str] = None):
        self.cam_id = cam_id
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        self.copilot = copilot
        self.stride = max(1, stride)
        self.realtime = realtime
        self.loop = loop
        self.on_alert, self.on_fault, self.on_clip, self.on_frame = on_alert, on_fault, on_clip, on_frame
        self.clip = clip
        self.stale_s = stale_s
        self.sensors_csv = sensors_csv
        self.health = Health()
        self.tamper = TamperDetector()
        self.stop_ev = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{cam_id}")
        self.latest_jpg: bytes = b""
        self.latency_ms: Deque[float] = collections.deque(maxlen=300)
        self.paused_until = 0.0  # human override: analysis paused (stream still monitored)
        self.paused_by = ""
        self.state: dict = {}
        self._lock = threading.Lock()

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str) and not str(self.source).startswith(("rtsp://", "rtsps://", "http://", "https://"))

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_ev.set()
        self.thread.join(timeout)
        self.health.state = "stopped"

    def pause(self, seconds: float, by: str) -> None:
        """Stop AI analysis on this camera for a bounded time (maintenance, drills, events)."""
        self.paused_until = time.time() + max(0.0, seconds)
        self.paused_by = by

    def resume(self) -> None:
        self.paused_until = 0.0
        self.paused_by = ""

    @property
    def paused(self) -> bool:
        return time.time() < self.paused_until

    def status(self) -> dict:
        h = self.health
        age = time.time() - h.last_frame_wall if h.last_frame_wall else None
        if h.state == "live" and age is not None and age > self.stale_s:
            h.state = "stale"
        return {"id": self.cam_id, "zone": self.copilot.zone, "health": h.to_dict(), "state": self.state,
                "paused": self.paused, "paused_until": self.paused_until if self.paused else None,
                "paused_by": self.paused_by if self.paused else ""}

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if not self.is_file:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process the newest frame from live streams
        return cap if cap.isOpened() else None

    def _run(self) -> None:
        backoff = 1.0
        while not self.stop_ev.is_set():
            cap = self._open()
            if cap is None:
                self.health.state = "offline"
                self.health.last_error = redact(f"cannot open {self.source!r}")
                if self.on_fault:
                    self.on_fault(self.cam_id, "offline", self.health.last_error)
                self.stop_ev.wait(backoff)
                backoff = min(30.0, backoff * 2)
                self.health.reconnects += 1
                continue
            backoff = 1.0
            self.health.state = "live"
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fps = fps if 1 < fps < 241 else 25.0
            if self.is_file:
                self.copilot.reset()
                if self.sensors_csv and os.path.exists(self.sensors_csv):
                    self.copilot.sensors.load_csv(self.sensors_csv)
            idx, t0, last = 0, time.perf_counter(), time.perf_counter()
            mono0 = time.monotonic()
            while not self.stop_ev.is_set():
                if self.is_file and self.realtime:
                    # behave like a live camera under load: skip frames we are too slow for
                    behind = int((time.perf_counter() - t0) * fps) - idx
                    for _ in range(max(0, behind - 1)):
                        if not cap.grab():
                            break
                        idx += 1
                        self.health.dropped = getattr(self.health, "dropped", 0) + 1
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                idx += 1
                if (idx - 1) % self.stride:
                    continue
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                # files: stream time (reproducible); live: wall clock, the same clock that
                # stamps MQTT/HTTP sensor readings, so camera and sensor evidence line up
                t = (idx - 1) / fps if self.is_file else time.monotonic() - mono0
                wall = t if self.is_file else time.time()
                if self.is_file and self.realtime:
                    lag = t - (time.perf_counter() - t0)
                    if lag > 0:
                        self.stop_ev.wait(lag)
                try:
                    self._step(frame, t, wall)
                except Exception as e:  # noqa: BLE001 - one bad frame must never kill a camera thread
                    self.health.last_error = redact(f"{type(e).__name__}: {e}")[:300]
                    self.health.errors = getattr(self.health, "errors", 0) + 1
                    log.exception("camera %s: frame processing failed", self.cam_id)
                    if self.health.errors in (1, 10, 100) and self.on_fault:
                        self.on_fault(self.cam_id, "processing_error", self.health.last_error)
                now = time.perf_counter()
                self.health.fps = 0.9 * self.health.fps + 0.1 / max(now - last, 1e-6)
                last = now
            cap.release()
            if self.stop_ev.is_set():
                break
            if self.is_file and not self.loop:
                self.health.state = "stopped"
                break
            self.health.state = "connecting"
            self.health.reconnects += 1
            log.warning("camera %s: stream ended, reconnecting in %.0fs", self.cam_id, backoff)
            self.stop_ev.wait(backoff)

    def _step(self, frame: np.ndarray, t: float, wall: Optional[float] = None) -> None:
        wall = t if wall is None else wall
        h = self.health
        h.frames += 1
        h.last_frame_wall = time.time()
        if h.state != "live":
            h.state = "live"
        faults = self.tamper(frame, t)
        for k, v in faults.items():
            if v and not h.faults.get(k) and self.on_fault:
                self.on_fault(self.cam_id, k, f"camera {self.cam_id}: {k} detected")
            h.faults[k] = v
        cop = self.copilot
        cop.sensors.advance(wall)
        if self.paused:
            img = frame.copy()
            cv2.putText(img, f"AI ANALYSIS PAUSED by {self.paused_by}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, f"AI ANALYSIS PAUSED by {self.paused_by}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
            safe = anonymize(img, [], "raw" if cop.privacy == "raw" else "skeleton")
            ok, buf = cv2.imencode(".jpg", safe, [cv2.IMWRITE_JPEG_QUALITY, 60])
            with self._lock:
                if ok:
                    self.latest_jpg = buf.tobytes()
                self.state = {"t": round(t, 2), "paused": True}
            if self.on_frame:
                self.on_frame(self.cam_id, self.latest_jpg, self.state)
            return
        res = cop.process(frame, t, wall_t=wall)
        if res.scene_change and not self.is_file and self.on_fault:
            # a live camera's field of view changed (moved, knocked, re-aimed): calibration no longer applies
            self.on_fault(self.cam_id, "view_changed", f"camera {self.cam_id}: field of view changed - re-calibrate")
        self.latency_ms.append(res.timings.get("total", 0.0))
        view = cop._last_frame if cop._last_frame is not None else frame
        img = cop.render(view, res.people, res.hazards, res.assessments)
        if self.clip is not None:
            for alert_id, frames in self.clip.add(anonymize(view, res.privacy_people, cop.privacy,
                                                            pose_available=cop.pose is not None), t):
                if self.on_clip:
                    self.on_clip(self.cam_id, alert_id, frames)
        for al in res.new_alerts:
            if self.clip is not None:
                self.clip.mark(al.id, t)
            if self.on_alert:
                self.on_alert(self.cam_id, al)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
        gauges: Dict[str, dict] = {}
        for a in res.assessments:
            g = gauges.get(a.hazard)
            if g is None or a.prob > g["prob"]:
                gauges[a.hazard] = {"prob": round(a.prob, 3), "subject": a.subject, "active": a.active,
                                    "threshold": a.threshold,
                                    "top": [{"name": c.name, "c": round(c.contribution, 2), "v": round(c.value, 3)}
                                            for c in a.top_cues(3, positive_only=False, min_abs=0.0)]}
        with self._lock:
            if ok:
                self.latest_jpg = buf.tobytes()
            self.state = {"t": round(t, 2), "people": len(res.people), "motion": round(res.motion, 4),
                          "ran_detector": res.ran_detector, "timings": {k: round(v, 1) for k, v in res.timings.items()},
                          "fps": round(h.fps, 1), "gauges": gauges, "sensors": {z: v for z, v in cop.sensors.status(wall).items() if z == cop.zone or (self.is_file and z == "default")},
                          "scene_change": res.scene_change}
        if self.on_frame:
            self.on_frame(self.cam_id, self.latest_jpg, self.state)
