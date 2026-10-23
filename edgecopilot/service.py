"""Service wiring: cameras + shared sensors + durable store + notifications.

:class:`EdgeService` is the production runtime behind the API server.  It
turns a validated config into running components and owns the alert path::

    camera thread -> Copilot.process -> new Alert
        -> store (SQLite + encrypted media, provenance) -> audit
        -> notifier (escalation)                         -> live broadcast
        -> clip buffer (pre/post-event MP4, encrypted)

Operator actions (confirm / dismiss / notes) flow back into the copilot's
online calibration and into the audit trail.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from . import __version__
from .cameras import CameraWorker, ClipBuffer, redact
from .copilot.events import EventLog
from .io.sensors import SensorHub
from .notify import Notifier
from .pipeline import Copilot
from .storage import Store

log = logging.getLogger("edgecopilot.service")


FAULT_TEXT = {
    "offline": "camera offline / stream unreachable",
    "frozen": "frozen image (stream stuck or replayed still)",
    "blackout": "image black (lens covered, power or IR failure)",
    "defocus": "image lost its detail (lens obstructed, sprayed or moved)",
    "view_changed": "field of view changed (camera moved) - calibration no longer valid",
    "processing_error": "analysis software error on this camera's frames",
}


def utc_iso(ts: Optional[float]) -> Optional[str]:
    """ISO-8601 UTC timestamp (records are kept in UTC; displays convert to local time)."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sd_notify(msg: str) -> bool:
    """systemd notify protocol (READY=1 / WATCHDOG=1) without extra dependencies."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.send(msg.encode())
        return True
    except OSError:
        return False


def file_sha256(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class EdgeService:
    def __init__(self, cfg: dict, root: str = ".", store: Optional[Store] = None, start_mqtt: bool = True,
                 audit_key: Optional[bytes] = None):
        self.cfg = cfg
        self.root = root
        st = cfg.get("storage", {})
        self.store = store or Store(os.path.join(root, st.get("db_path", "data/edgecopilot.db")),
                                    os.path.join(root, st.get("media_dir", "data/media")),
                                    st.get("encrypt_media", True), st.get("key_env", "EDGECOPILOT_MEDIA_KEY"),
                                    audit_key=audit_key)
        self.retention_days = st.get("retention_days", 30)
        # live sensors: only configured zones, never borrowed across rooms
        self.sensors = SensorHub(zones=list(cfg.get("zones", {})), fallback_zone=None)
        self.cameras: Dict[str, CameraWorker] = {}
        self.copilots: Dict[str, Copilot] = {}
        self.listeners: List[Callable[[dict], None]] = []
        self.lock = threading.RLock()
        self.started = time.time()
        self.provenance = {
            "version": __version__,
            "config_sha256": hashlib.sha256(repr(sorted(cfg.get("hazards", {}).items())).encode()).hexdigest()[:16],
            "models": {k: (file_sha256(self._abs(v["path"])) or "missing")[:16] for k, v in cfg["models"].items()},
        }
        self.mqtt = None
        mq = cfg.get("mqtt")
        if mq and start_mqtt:
            try:
                from .io.mqtt import MqttBridge
                self.mqtt = MqttBridge.from_config(mq, self.sensors.push)
                self.mqtt.start()
            except Exception as e:  # noqa: BLE001 - MQTT is optional, HTTP ingestion still works
                log.error("MQTT disabled: %s", e)
        self.notifier = Notifier.from_config(
            cfg.get("notifications") or {}, self.is_open, self.store.audit,
            mqtt_publish=self.mqtt.publish if self.mqtt else None,
            base_url=os.environ.get("EDGECOPILOT_PUBLIC_URL", ""))
        self._retention_stop = threading.Event()
        self._retention = threading.Thread(target=self._retention_loop, daemon=True, name="retention")
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True, name="watchdog")
        self._fault_last: Dict[str, float] = {}
        self.fault_cooldown_s = 900.0

    def _abs(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    # ----------------------------------------------------------- cameras
    def add_camera(self, cam: dict, sensors_csv: Optional[str] = None) -> CameraWorker:
        cfg = dict(self.cfg)
        if cam.get("privacy"):
            cfg = {**cfg, "runtime": {**cfg["runtime"], "privacy": cam["privacy"]}}
        # a recorded video with its own sensor CSV replays into a private hub (never mixes with live sensors)
        hub = SensorHub(fallback_zone="default") if sensors_csv else self.sensors
        cop = Copilot(cfg, root=self.root, zone=cam["zone"], sensors=hub, event_log=EventLog(max_items=200),
                      enable=cam.get("hazards") or None)
        cop.status_listeners.append(self._on_status)
        st = self.cfg.get("storage", {})
        # ${VAR} in camera URLs is filled from the environment so credentials never sit in the config file
        source = os.path.expandvars(str(cam["source"]))
        w = CameraWorker(cam["id"], source, cop, stride=cam.get("stride", 1), realtime=cam.get("realtime", True),
                         loop=cam.get("loop", False), on_alert=self._on_alert, on_fault=self._on_fault,
                         on_clip=self._on_clip, on_frame=self._on_frame,
                         clip=ClipBuffer(st.get("clip_pre_s", 5.0), st.get("clip_post_s", 10.0)),
                         sensors_csv=sensors_csv)
        with self.lock:
            self.cameras[cam["id"]] = w
            self.copilots[cam["id"]] = cop
        return w

    def start(self) -> None:
        for w in self.cameras.values():
            w.start()
        self._retention.start()
        self._watchdog.start()
        self.store.audit("system", "admin", "service_start", "", {"cameras": list(self.cameras), **self.provenance})
        sd_notify("READY=1")

    def stop(self) -> None:
        for w in self.cameras.values():
            w.stop()
        self._retention_stop.set()
        self.notifier.stop()
        if self.mqtt:
            self.mqtt.stop()
        self.store.audit("system", "admin", "service_stop", "", {})

    def healthy(self) -> bool:
        """Liveness for the watchdog: every enabled camera thread alive and producing frames (or paused)."""
        for w in self.cameras.values():
            if w.health.state == "stopped":
                continue
            if not w.thread.is_alive():
                return False
        return True

    def liveness(self, max_frame_age_s: float = 30.0) -> tuple:
        """(ok, problem). Dead camera threads or a live stream that stopped delivering frames = not alive.
        Cameras that are offline/reconnecting are a *camera* problem (trouble alert), not a reason to restart."""
        now = time.time()
        for cid, w in self.cameras.items():
            st = w.health.state
            if st == "stopped":
                continue
            if not w.thread.is_alive():
                return False, f"camera thread {cid} died"
            if st == "live" and w.health.last_frame_wall and now - w.health.last_frame_wall > max_frame_age_s and not w.paused:
                return False, f"camera {cid} stalled ({now - w.health.last_frame_wall:.0f} s without a processed frame)"
        return True, ""

    def _watchdog_loop(self) -> None:
        """Kick the systemd watchdog only while healthy: a hung pipeline gets restarted automatically."""
        while not self._retention_stop.wait(5.0):
            if self.healthy() and self.liveness()[0]:
                sd_notify("WATCHDOG=1")
            else:
                log.error("watchdog: a camera thread died - withholding WATCHDOG=1 so the service is restarted")

    def _retention_loop(self) -> None:
        while True:  # purge at start-up, then hourly
            try:
                self.store.purge(self.retention_days)
            except Exception as e:  # noqa: BLE001
                log.error("retention purge failed: %s", e)
            if self._retention_stop.wait(3600.0):
                return

    # -------------------------------------------------------- alert path
    def _broadcast(self, msg: dict) -> None:
        for cb in list(self.listeners):
            try:
                cb(msg)
            except Exception:  # noqa: BLE001 - a broken listener must never stop the camera thread
                log.exception("listener failed")

    def _on_frame(self, cam_id: str, jpg: bytes, state: dict) -> None:
        if self.listeners:
            self._broadcast({"type": "frame", "camera": cam_id, "jpg": jpg, "state": state})

    def _on_alert(self, cam_id: str, al) -> None:
        rec = al.summary()
        rec["camera_id"] = cam_id
        rec["triage"] = al.triage.get("mode") if isinstance(al.triage, dict) else al.triage
        rec["provenance"] = {**self.provenance, "triage": al.triage}
        try:
            if al.snapshot_jpg_b64:
                rec["snapshot_path"] = self.store.put_media(f"{al.id}_snapshot.jpg", base64.b64decode(al.snapshot_jpg_b64))
            if al.evidence_jpg_b64:
                rec["evidence_path"] = self.store.put_media(f"{al.id}_evidence.jpg", base64.b64decode(al.evidence_jpg_b64))
            self.store.save_alert(rec)
            self.store.audit("system", "admin", "alert_raised", al.id,
                             {"camera": cam_id, "hazard": al.hazard, "prob": round(al.prob, 3), "severity": al.severity,
                              "triage": rec["triage"], "grounded": al.verification.get("groundedness")})
        except Exception:  # noqa: BLE001 - storage failure must not suppress the live alert
            log.exception("failed to persist alert %s", al.id)
        log.warning("ALERT %s %s %s p=%.2f", cam_id, al.hazard, al.severity, al.prob,
                    extra={"camera": cam_id, "alert_id": al.id, "hazard": al.hazard, "event": "alert"})
        # the images now live encrypted in the store; drop the in-memory copies (bounded memory on long runs)
        al.snapshot_jpg_b64 = ""
        al.evidence_jpg_b64 = ""
        self.notifier.alert(rec)
        self._broadcast({"type": "alert", "alert": rec})

    def _on_status(self, al, change: str) -> None:
        """Alarms latch: an alert that clears by itself stays 'open' (and keeps escalating) until a person
        acknowledges it, but the clearing time and the peak score are recorded."""
        try:
            if change == "cleared":
                self.store.update_alert(al.id, cleared_at=time.time(), peak_prob=float(al.peak_prob))
            elif change == "reopened":
                self.store.update_alert(al.id, cleared_at=None, peak_prob=float(al.peak_prob))
            self._broadcast({"type": "alert_update", "id": al.id, "change": change})
        except Exception:  # noqa: BLE001
            log.exception("failed to persist status change of %s", al.id)

    def _on_fault(self, cam_id: str, kind: str, detail: str) -> None:
        """Trouble signal: a failing camera is itself an incident (FM 3232-style), routed like an alert."""
        detail = redact(detail)
        key = f"{cam_id}:{kind}"
        now = time.time()
        self._broadcast({"type": "camera_fault", "camera": cam_id, "kind": kind, "detail": detail})
        if now - self._fault_last.get(key, 0.0) < self.fault_cooldown_s:
            return  # already reported recently (reconnect loops must not flood operators)
        self._fault_last[key] = now
        w = self.cameras.get(cam_id)
        zone = w.copilot.zone if w else ""
        text = FAULT_TEXT.get(kind, kind)
        rec = {"id": uuid.uuid4().hex[:10], "camera_id": cam_id, "zone": zone, "hazard": "camera_fault", "subject": kind,
               "t_video": 0.0, "wall_time": now, "prob": 1.0, "peak_prob": 1.0, "severity": "HIGH", "triage": "TROUBLE",
               "status": "open",
               "explanation": {"headline": f"Camera {cam_id} ({zone}): {text}", "because": detail,
                               "text": f"Camera {cam_id} in {zone}: {text}. This area is NOT being monitored until it is fixed.",
                               "confidence_text": "", "counterfactual": ""},
               "decision": {}, "verification": {"groundedness": 1.0, "n_claims": 0},
               "recommendations": [{"text": f"Inspect camera {cam_id} in {zone} and restore the view", "priority": 1},
                                   {"text": f"Arrange manual patrols of {zone} until monitoring is restored", "priority": 1}],
               "cues": [], "provenance": self.provenance}
        try:
            self.store.save_alert(rec)
            self.store.audit("system", "admin", "camera_fault", cam_id, {"kind": kind, "detail": detail, "alert": rec["id"]})
        except Exception:  # noqa: BLE001
            log.exception("failed to persist camera fault")
        log.error("camera fault %s: %s", cam_id, detail, extra={"camera": cam_id, "event": "camera_fault"})
        self.notifier.alert(rec)
        self._broadcast({"type": "alert", "alert": rec})

    def _on_clip(self, cam_id: str, alert_id: str, frames) -> None:
        def work():
            data = ClipBuffer.encode_mp4(frames)
            if data:
                p = self.store.put_media(f"{alert_id}_clip.mp4", data)
                self.store.update_alert(alert_id, clip_path=p)
        threading.Thread(target=work, daemon=True, name=f"clip-{alert_id}").start()

    # ---------------------------------------------------- operator actions
    def is_open(self, alert_id: str) -> bool:
        a = self.store.get_alert(alert_id)
        return a is not None and a["status"] == "open"

    def feedback(self, alert_id: str, confirmed: bool, actor: str, role: str, note: str = "") -> Optional[dict]:
        a = self.store.get_alert(alert_id)
        if a is None:
            return None
        if a["status"] not in ("open", "resolved"):
            raise ValueError(f"alert already {a['status']} by {a.get('operator') or 'someone'}")
        status = "confirmed" if confirmed else "dismissed"
        self.store.update_alert(alert_id, status=status, operator=actor, operator_note=note[:500], ack_time=time.time())
        cop = self.copilots.get(a.get("camera_id") or "")
        out = {"id": alert_id, "status": status}
        if cop is not None and alert_id in cop.events.alerts and a["hazard"] in cop.models:
            r = cop.feedback(alert_id, confirmed, note)
            if r:
                out.update(r)
        self.store.audit(actor, role, f"alert_{status}", alert_id, {"note": note[:200], **{k: v for k, v in out.items() if k != 'id'}})
        self._broadcast({"type": "alert_update", "id": alert_id, "status": status, "operator": actor})
        return out

    # -------------------------------------------------- human oversight
    def pause_camera(self, cam_id: str, seconds: float, actor: str, role: str, reason: str = "") -> Optional[dict]:
        w = self.cameras.get(cam_id)
        if w is None:
            return None
        w.pause(seconds, actor)
        self.store.audit(actor, role, "camera_paused", cam_id, {"seconds": seconds, "reason": reason[:200]})
        self._broadcast({"type": "camera_paused", "camera": cam_id, "by": actor, "until": w.paused_until})
        return w.status()

    def resume_camera(self, cam_id: str, actor: str, role: str) -> Optional[dict]:
        w = self.cameras.get(cam_id)
        if w is None:
            return None
        w.resume()
        self.store.audit(actor, role, "camera_resumed", cam_id, {})
        self._broadcast({"type": "camera_resumed", "camera": cam_id, "by": actor})
        return w.status()

    def status(self) -> dict:
        return {"version": __version__, "uptime_s": round(time.time() - self.started, 1), "now_utc": utc_iso(time.time()),
                "cameras": [w.status() for w in self.cameras.values()],
                "sensors": self.sensors.status(time.time()),
                "mqtt": {"connected": bool(self.mqtt and self.mqtt.connected.is_set()),
                         "received": getattr(self.mqtt, "received", 0), "rejected": getattr(self.mqtt, "rejected", 0)}
                if self.mqtt else None,
                "provenance": self.provenance}
