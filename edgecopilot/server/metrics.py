"""Prometheus metrics (own registry) refreshed from the running service every 2 s."""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

REG = CollectorRegistry()
UP = Gauge("edgecopilot_camera_up", "1 if the camera stream is live", ["camera", "zone"], registry=REG)
FPS = Gauge("edgecopilot_camera_fps", "Processed frames per second", ["camera"], registry=REG)
LAT = Gauge("edgecopilot_frame_latency_ms", "Per-frame processing latency", ["camera", "quantile"], registry=REG)
RISK = Gauge("edgecopilot_risk", "Current hazard probability", ["camera", "hazard"], registry=REG)
FAULT = Gauge("edgecopilot_camera_fault", "Camera fault active (frozen/blackout/defocus)", ["camera", "kind"], registry=REG)
PEOPLE = Gauge("edgecopilot_people", "People in view", ["camera"], registry=REG)
RECONNECTS = Gauge("edgecopilot_camera_reconnects", "Stream reconnects since start", ["camera"], registry=REG)
ALERTS = Counter("edgecopilot_alerts_total", "Alerts raised", ["camera", "hazard", "severity", "triage"], registry=REG)
FEEDBACK = Counter("edgecopilot_operator_feedback_total", "Operator decisions", ["outcome"], registry=REG)
NOTIFY = Counter("edgecopilot_notifications_total", "Notification deliveries", ["ok"], registry=REG)
SENSOR_AGE = Gauge("edgecopilot_sensor_age_seconds", "Age of the latest sensor reading", ["zone", "sensor"], registry=REG)

_state = {"ok": 0, "fail": 0}


def _on_event(msg: dict) -> None:
    if msg.get("type") == "alert":
        a = msg["alert"]
        ALERTS.labels(a.get("camera_id", "?"), a.get("hazard", "?"), a.get("severity", "?"), a.get("triage", "?")).inc()


def refresh(service) -> None:
    for w in list(service.cameras.values()):
        st = w.status()
        h = st["health"]
        UP.labels(w.cam_id, st["zone"]).set(1.0 if h["state"] == "live" else 0.0)
        FPS.labels(w.cam_id).set(h["fps"])
        RECONNECTS.labels(w.cam_id).set(h["reconnects"])
        for k in ("frozen", "blackout", "defocus"):
            FAULT.labels(w.cam_id, k).set(1.0 if h["faults"].get(k) else 0.0)
        lat = list(w.latency_ms)
        if lat:
            for q in (0.5, 0.95, 0.99):
                LAT.labels(w.cam_id, str(q)).set(float(np.quantile(lat, q)))
        state = st.get("state") or {}
        PEOPLE.labels(w.cam_id).set(state.get("people", 0))
        for hz, g in (state.get("gauges") or {}).items():
            RISK.labels(w.cam_id, hz).set(g["prob"])
    n = service.notifier
    if n.ok_count > _state["ok"]:
        NOTIFY.labels("true").inc(n.ok_count - _state["ok"])
        _state["ok"] = n.ok_count
    if n.fail_count > _state["fail"]:
        NOTIFY.labels("false").inc(n.fail_count - _state["fail"])
        _state["fail"] = n.fail_count
    for zone, fields in service.sensors.status(time.time()).items():
        for k, v in fields.items():
            SENSOR_AGE.labels(zone, k).set(v["age_s"])


def start_collector(service, period: float = 2.0) -> None:
    service.listeners.append(_on_event)

    def loop():
        while True:
            try:
                refresh(service)
            except Exception:  # noqa: BLE001 - metrics must never take the service down
                logging.getLogger("edgecopilot.metrics").exception("metrics refresh failed")
            time.sleep(period)
    threading.Thread(target=loop, daemon=True, name="metrics").start()


def render() -> bytes:
    return generate_latest(REG)
