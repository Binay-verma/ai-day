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
import math
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from . import __version__
from .cameras import CameraWorker, ClipBuffer, redact
from .copilot.events import EventLog
from .io.sensors import SensorHub
from .notify import Notifier
from .track_record import TrackRecord
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
    "self_test_failed": "periodic self-test failed (models, frames or alert chain)",
    "low_visibility": "image too dark, flat or noisy for reliable analysis (night without IR, fog, glare, sensor noise)",
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
        self.track_record = TrackRecord(self.store)
        self.notifier = Notifier.from_config(
            cfg.get("notifications") or {}, self.is_open, self.store.audit,
            mqtt_publish=self.mqtt.publish if self.mqtt else None,
            base_url=os.environ.get("EDGECOPILOT_PUBLIC_URL", ""))
        self._retention_stop = threading.Event()
        self._retention = threading.Thread(target=self._retention_loop, daemon=True, name="retention")
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True, name="watchdog")
        self._fault_last: Dict[str, float] = {}
        self.last_self_test: Optional[dict] = None
        self.self_test_period_s = float(cfg.get("runtime", {}).get("self_test_minutes", 60)) * 60
        self.fault_cooldown_s = 900.0

    def _abs(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    # ----------------------------------------------------------- cameras
    def add_camera(self, cam: dict, sensors_csv: Optional[str] = None) -> CameraWorker:
        cfg = dict(self.cfg)
        if cam.get("pose"):  # per-camera pose model (accuracy where falls matter most, speed elsewhere)
            cfg = {**cfg, "models": {**cfg["models"], "pose": {**cfg["models"]["pose"], **{k: v for k, v in cam["pose"].items() if v is not None}}}}
        if cam.get("privacy"):
            cfg = {**cfg, "runtime": {**cfg["runtime"], "privacy": cam["privacy"]}}
        # a recorded video with its own sensor CSV replays into a private hub (never mixes with live sensors)
        hub = SensorHub(fallback_zone="default") if sensors_csv else self.sensors
        cop = Copilot(cfg, root=self.root, zone=cam["zone"], sensors=hub, event_log=EventLog(max_items=200),
                      enable=cam.get("hazards") or None, regions=cam.get("restricted") or None,
                      masks=cam.get("masks") or None)
        cop.status_listeners.append(self._on_status)
        st = self.cfg.get("storage", {})
        # ${VAR} in camera URLs is filled from the environment so credentials never sit in the config file
        source = os.path.expandvars(str(cam["source"]))
        w = CameraWorker(cam["id"], source, cop, stride=cam.get("stride", 1), realtime=cam.get("realtime", True),
                         loop=cam.get("loop", False), on_alert=self._on_alert, on_fault=self._on_fault,
                         on_clip=self._on_clip, on_frame=self._on_frame,
                         clip=ClipBuffer(st.get("clip_pre_s", 5.0), st.get("clip_post_s", 10.0)),
                         sensors_csv=sensors_csv)
        w.on_held = self._on_held
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

    # ---------------------------------------------------------- self-test
    def self_test(self, notify: bool = False, actor: str = "system") -> dict:
        """End-to-end supervision (FM 3232 / NFPA 72 style "prove it still works").

        Per camera: frames are fresh (or the camera is deliberately paused); the person
        and fire models run on the latest frame within 5 s; a synthetic hazard goes
        through the real fusion -> explanation -> verifier -> playbook chain and must
        come out grounded with at least one action. Also checks the audit store. Any
        failing camera raises a ``self_test_failed`` trouble alert. With ``notify`` a
        clearly marked TEST message is sent through every notification channel.
        """
        import cv2
        import numpy as np

        from .xai.nlg import explain, numbers_in, verify
        t0 = time.time()
        res: Dict[str, Any] = {"time_utc": utc_iso(t0), "_t": t0, "cameras": {}, "ok": True}
        for cid, w in self.cameras.items():
            checks: Dict[str, Any] = {}
            cop = w.copilot
            age = t0 - w.health.last_frame_wall if w.health.last_frame_wall else None
            checks["frames"] = {"ok": bool(w.paused or (age is not None and age < 30)), "age_s": None if age is None else round(age, 1),
                                "paused": bool(w.paused)}
            frame = None
            if w.latest_jpg:
                frame = cv2.imdecode(np.frombuffer(w.latest_jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                frame = np.full((360, 640, 3), 128, np.uint8)
            for name, m in (("pose_model", cop.pose), ("fire_model", cop.fire)):
                if m is None:  # degraded: that hazard runs on weaker cues only
                    checks[name] = {"ok": False, "detail": "model not loaded"}
                    continue
                try:
                    s = time.perf_counter()
                    m(frame)
                    ms = (time.perf_counter() - s) * 1e3
                    checks[name] = {"ok": ms < 5000, "ms": round(ms, 1)}
                except Exception as e:  # noqa: BLE001
                    checks[name] = {"ok": False, "detail": f"{type(e).__name__}: {e}"[:200]}
            try:
                chain_ok, detail = True, []
                for hz, model in cop.models.items():
                    vals = {n: (sp.hi if sp.weight > 0 else sp.lo) for n, sp in model.specs.items()}
                    a = model.assess(cop.zone, 0.0, vals)
                    rec = cop.recommender.recommend(a, cop.zone)
                    texts = [r["text"] for r in rec["actions"]]
                    expl = explain(a, model, cop.zone, texts)
                    v = verify(expl.text, a, model, cop.zone, extra_numbers=numbers_in(texts), recommendations=texts)
                    good = a.prob >= model.threshold and v.ok and bool(texts) and a.attribution_error() < 1e-9
                    chain_ok &= good
                    detail.append(f"{hz}:{'ok' if good else 'FAIL'}")
                checks["alert_chain"] = {"ok": chain_ok, "detail": " ".join(detail)}
            except Exception as e:  # noqa: BLE001
                checks["alert_chain"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"[:200]}
            cam_ok = all(c["ok"] for c in checks.values())
            res["cameras"][cid] = {"ok": cam_ok, "checks": checks}
            if not cam_ok:
                res["ok"] = False
                bad = ", ".join(f"{k} ({v.get('detail', v)})" for k, v in checks.items() if not v["ok"])
                self._on_fault(cid, "self_test_failed", f"camera {cid}: self-test failed: {bad}")
        try:
            self.store.audit(actor, "system" if actor == "system" else "admin", "self_test", "", {"ok": res["ok"], "notify": notify})
            res["store"] = {"ok": True}
        except Exception as e:  # noqa: BLE001
            res["store"] = {"ok": False, "detail": str(e)[:200]}
            res["ok"] = False
        res["notification_channels"] = len(getattr(self.notifier, "channels", {}) or {})
        if notify:
            rec = {"id": "test-" + uuid.uuid4().hex[:8], "camera_id": "-", "zone": "-", "hazard": "self_test", "subject": "test",
                   "t_video": 0.0, "wall_time": t0, "prob": 0.0, "peak_prob": 0.0, "severity": "INFO", "triage": "TEST",
                   "status": "resolved",
                   "explanation": {"headline": "TEST message from the edge copilot - no action required",
                                   "because": f"self-test requested by {actor}",
                                   "text": "TEST - this is a scheduled/manual test of the alert notification chain. No action required.",
                                   "confidence_text": "", "counterfactual": ""},
                   "decision": {}, "verification": {"groundedness": 1.0, "n_claims": 0}, "recommendations": [], "cues": []}
            self.notifier.alert(rec)
            res["test_notification"] = rec["id"]
        res["duration_ms"] = round((time.time() - t0) * 1e3, 1)
        self.last_self_test = res
        return res

    def _watchdog_loop(self) -> None:
        """Kick the systemd watchdog only while healthy: a hung pipeline gets restarted automatically.
        Also runs the periodic end-to-end self-test."""
        next_test = time.time() + min(120.0, self.self_test_period_s or 120.0)
        while not self._retention_stop.wait(5.0):
            if self.self_test_period_s > 0 and time.time() >= next_test:
                next_test = time.time() + self.self_test_period_s
                try:
                    self.self_test()
                except Exception:  # noqa: BLE001
                    log.exception("self-test crashed")
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
            elif change == "long_lie":
                self.store.update_alert(al.id, severity="CRITICAL", triage="ALARM")
                self.store.audit("system", "system", "long_lie_escalation", al.id, {"zone": al.zone, "subject": al.subject})
                rec = self.store.get_alert(al.id) or {}
                if rec:
                    rec = {**rec, "severity": "CRITICAL", "triage": "ALARM",
                           "explanation": {**(rec.get("explanation") or {}),
                                           "headline": f"STILL DOWN: {al.subject} in {al.zone} has not got up after the fall",
                                           "text": f"Escalation: {al.subject} in {al.zone} is still on the ground "
                                                   "a long time after the fall alert. "
                                                   + (rec.get("explanation") or {}).get("text", "")}}
                    self.notifier.alert(rec)
            self._broadcast({"type": "alert_update", "id": al.id, "change": change})
        except Exception:  # noqa: BLE001
            log.exception("failed to persist status change of %s", al.id)

    def _on_held(self, cam_id: str, a, gate) -> None:
        """An evidence gate held a would-be alert below the line (e.g. someone lying down, not falling).

        Not an alert and nothing is paged: it is written to the audit trail (rate-limited per person)
        so that commissioning tests and reviews can see that the gate, not a blind spot, kept it quiet.
        """
        key = f"held:{cam_id}:{a.hazard}:{a.subject}"
        now = time.time()
        if now - self._fault_last.get(key, 0.0) < 20.0:
            return
        self._fault_last[key] = now
        prob_without_gate = 1.0 / (1.0 + math.exp(-max(min(a.logit - gate.contribution, 50.0), -50.0)))
        detail = {"hazard": a.hazard, "subject": a.subject, "t_video": round(float(a.t), 2), "gate": gate.name,
                  "reason": gate.phrase, "prob_shown": round(a.prob, 3), "prob_without_gate": round(prob_without_gate, 3),
                  "threshold": a.threshold,
                  "cues": [{"name": c.name, "value": round(c.value, 3), "c": round(c.contribution, 2)} for c in a.cues]}
        self._broadcast({"type": "held", "camera": cam_id, **{k: detail[k] for k in ("hazard", "subject", "reason")}})
        try:
            self.store.audit("system", "admin", "hazard_held", cam_id, detail)
        except Exception:  # noqa: BLE001
            log.exception("could not record held event")

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

    def what_if(self, alert_id: str, changes: Optional[Dict[str, float]] = None) -> Optional[dict]:
        """Re-score a stored alert with some cue values changed (operator "what if?" tool).

        Exact, because the hazard model is additive: the new log-odds is the alert's own
        log-odds plus, for each changed cue, reliability x (f(new) - f(old)) with the
        camera's current cue function f. Read-only: nothing about the alert changes.
        """
        import math
        a = self.store.get_alert(alert_id)
        if a is None:
            return None
        cop = self.copilots.get(a.get("camera_id") or "")
        model = cop.models.get(a.get("hazard", "")) if cop else None
        if model is None:
            raise ValueError("no hazard model for this alert (trouble alerts have no ledger)")
        p = min(max(float(a.get("prob", 0.5)), 1e-9), 1 - 1e-9)
        logit0 = math.log(p / (1 - p))
        margin = float((a.get("decision") or {}).get("margin_logit", 0.0))
        tau = logit0 - margin
        changes = changes or {}
        cues, logit = [], logit0
        for c in a.get("cues") or []:
            spec = model.specs.get(c["name"])
            if spec is None:
                continue
            r = float(c.get("reliability", 1.0))
            old = float(c["value"])
            new = float(changes.get(c["name"], old))
            if not math.isfinite(new):
                raise ValueError(f"{c['name']}: value must be finite")
            new = min(max(new, min(spec.lo, spec.hi)), max(spec.lo, spec.hi))
            contrib_old = float(c["contribution"])
            contrib_new = contrib_old + r * spec.weight * (spec.transform(new) - spec.transform(old))
            logit += contrib_new - contrib_old
            cues.append({"name": c["name"], "unit": c.get("unit", ""), "value": old, "new_value": new,
                         "lo": min(spec.lo, spec.hi), "hi": max(spec.lo, spec.hi), "ref": spec.ref,
                         "reliability": r, "contribution": contrib_old, "new_contribution": contrib_new})
        prob = 1 / (1 + math.exp(-logit))
        thr = 1 / (1 + math.exp(-tau))
        return {"alert_id": alert_id, "hazard": a.get("hazard"), "threshold": thr, "prob_before": p,
                "prob": prob, "would_alert": prob >= thr, "cues": cues}

    def feedback(self, alert_id: str, confirmed: bool, actor: str, role: str, note: str = "") -> Optional[dict]:
        a = self.store.get_alert(alert_id)
        if a is None:
            return None
        if a["status"] not in ("open", "resolved"):
            raise ValueError(f"alert already {a['status']} by {a.get('operator') or 'someone'}")
        status = "confirmed" if confirmed else "dismissed"
        self.store.update_alert(alert_id, status=status, operator=actor, operator_note=note[:500], ack_time=time.time())
        self.track_record.invalidate()
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
