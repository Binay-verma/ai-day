"""Alert delivery: webhooks (generic / Slack / Teams), e-mail, MQTT - with
retries and time-based escalation.

Escalation policy (config ``notifications.escalation``)::

    - {after_s: 0,   channels: [security_slack]}          # immediately
    - {after_s: 60,  channels: [warden_email]}            # still unacknowledged after 1 min
    - {after_s: 180, channels: [principal_webhook, mqtt]} # still unacknowledged after 3 min

A level fires only if the alert is still open (not confirmed/dismissed/
resolved) when its time comes, so acknowledging on the console stops the
chain.  Deliveries run on a worker thread with exponential back-off, never on
the video thread, and every attempt is written to the audit log.
"""
from __future__ import annotations

import heapq
import json
import logging
import os
import smtplib
import ssl
import threading
import time
import urllib.request
from email.message import EmailMessage
from typing import Callable, Dict, List, Optional

log = logging.getLogger("edgecopilot.notify")
SEV_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def format_text(a: dict, base_url: str = "") -> str:
    exp = a.get("explanation", {})
    tri = a.get("triage") or ""
    tri = tri.get("mode", "") if isinstance(tri, dict) else str(tri)
    acts = "; ".join(r["text"] for r in (a.get("recommendations") or [])[:3])
    link = f"\nOpen: {base_url.rstrip('/')}/#alert-{a['id']}" if base_url else ""
    if a.get("hazard") == "fire_smoke":
        link += "\n(AI decision support - follow the building fire alarm and fire-safety procedures.)"
    return (f"[{a.get('severity')}{' / ' + tri if tri else ''}] {exp.get('headline', a.get('hazard'))}\n"
            f"{exp.get('because', '')}\n{exp.get('confidence_text', '')}\nDo: {acts}{link}")


class Channel:
    name = "base"

    def send(self, alert: dict, text: str) -> None:  # raises on failure
        raise NotImplementedError


class WebhookChannel(Channel):
    def __init__(self, name: str, url: str, kind: str = "generic", timeout: float = 5.0):
        if not url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError(f"webhook {name}: only https:// URLs (or localhost) are allowed")
        self.name, self.url, self.kind, self.timeout = name, url, kind, timeout

    def payload(self, alert: dict, text: str) -> dict:
        if self.kind in ("slack", "teams"):
            return {"text": text}
        keep = ("id", "hazard", "zone", "camera_id", "severity", "prob", "t_video", "wall_time", "triage",
                "explanation", "recommendations")
        return {"event": "edgecopilot.alert", "text": text, "alert": {k: alert.get(k) for k in keep}}

    def send(self, alert: dict, text: str) -> None:
        data = json.dumps(self.payload(alert, text), default=float).encode()
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"},  # noqa: S310
                                     method="POST")  # scheme restricted to https/localhost in __init__
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310  # nosec B310 - https/localhost enforced in __init__
            if r.status >= 300:
                raise RuntimeError(f"HTTP {r.status}")


class EmailChannel(Channel):
    name = "email"

    def __init__(self, host: str, port: int, sender: str, to: List[str], user_env: str, password_env: str,
                 starttls: bool = True):
        self.host, self.port, self.sender, self.to = host, port, sender, to
        self.user, self.password = os.environ.get(user_env), os.environ.get(password_env)
        self.starttls = starttls

    def send(self, alert: dict, text: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = text.splitlines()[0][:150]
        msg["From"], msg["To"] = self.sender, ", ".join(self.to)
        msg.set_content(text)
        with smtplib.SMTP(self.host, self.port, timeout=10) as s:
            if self.starttls:
                s.starttls(context=ssl.create_default_context())
            if self.user and self.password:
                s.login(self.user, self.password)
            s.send_message(msg)


class MqttChannel(Channel):
    name = "mqtt"

    def __init__(self, publish: Callable[[str, str], None], topic: str):
        self.publish, self.topic = publish, topic

    def send(self, alert: dict, text: str) -> None:
        keep = ("id", "hazard", "zone", "camera_id", "severity", "prob", "wall_time", "triage")
        self.publish(self.topic, json.dumps({"text": text, **{k: alert.get(k) for k in keep}}, default=float))


class Notifier:
    def __init__(self, channels: Dict[str, Channel], escalation: List[dict], is_open: Callable[[str], bool],
                 audit: Optional[Callable[..., None]] = None, max_retries: int = 5, base_url: str = "",
                 min_severity: Optional[Dict[str, str]] = None, backoff_s: float = 2.0):
        self.channels = channels
        self.levels = sorted(escalation, key=lambda x: x["after_s"]) or \
            [{"after_s": 0, "channels": list(channels)}]
        self.is_open = is_open
        self.audit = audit or (lambda *a, **k: None)
        self.max_retries = max_retries
        self.base_url = base_url
        self.min_sev = min_severity or {}
        self.backoff_s = backoff_s
        self._q: list = []  # heap of (due_time, seq, job)
        self._seq = 0
        self._cv = threading.Condition()
        self._stop = False
        self.sent: List[dict] = []  # bounded record for status/tests
        self.ok_count = 0
        self.fail_count = 0
        self._t = threading.Thread(target=self._run, daemon=True, name="notifier")
        self._t.start()

    @classmethod
    def from_config(cls, ncfg: dict, is_open, audit=None, mqtt_publish=None, base_url: str = "") -> Notifier:
        ch: Dict[str, Channel] = {}
        min_sev: Dict[str, str] = {}
        for name, w in (ncfg.get("webhooks") or {}).items():
            ch[name] = WebhookChannel(name, w["url"], w.get("kind", "generic"))
            min_sev[name] = w.get("min_severity", "MEDIUM")
        if ncfg.get("email"):
            e = ncfg["email"]
            ch["email"] = EmailChannel(e["host"], e.get("port", 587), e["sender"], e["to"], e.get("user_env", "SMTP_USER"),
                                       e.get("password_env", "SMTP_PASSWORD"), e.get("starttls", True))
            min_sev["email"] = e.get("min_severity", "HIGH")
        if ncfg.get("mqtt_topic") and mqtt_publish:
            ch["mqtt"] = MqttChannel(mqtt_publish, ncfg["mqtt_topic"])
        return cls(ch, ncfg.get("escalation") or [], is_open, audit, ncfg.get("max_retries", 5), base_url, min_sev)

    def _push(self, due: float, job: dict) -> None:
        with self._cv:
            self._seq += 1
            heapq.heappush(self._q, (due, self._seq, job))
            self._cv.notify()

    def alert(self, alert: dict) -> None:
        """Schedule all escalation levels for a new alert (non-blocking)."""
        now = time.time()
        for lvl_i, lvl in enumerate(self.levels):
            for ch in lvl["channels"]:
                if ch in self.channels:
                    self._push(now + lvl["after_s"], {"alert": alert, "channel": ch, "level": lvl_i, "attempt": 0})

    def stop(self, timeout: float = 2.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._t.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stop and (not self._q or self._q[0][0] > time.time()):
                    self._cv.wait(timeout=None if not self._q else max(0.0, self._q[0][0] - time.time()))
                if self._stop:
                    return
                _, _, job = heapq.heappop(self._q)
            self._deliver(job)

    def _deliver(self, job: dict) -> None:
        a, ch_name = job["alert"], job["channel"]
        if job["level"] > 0 and job["attempt"] == 0 and not self.is_open(a["id"]):
            return  # acknowledged in time: no escalation
        if SEV_RANK.get(a.get("severity", "LOW"), 0) < SEV_RANK.get(self.min_sev.get(ch_name, "LOW"), 0) and job["level"] == 0:
            return
        ch = self.channels[ch_name]
        try:
            ch.send(a, format_text(a, self.base_url))
            self.ok_count += 1
            rec = {"alert": a["id"], "channel": ch_name, "level": job["level"], "ok": True, "t": time.time()}
            self.audit("notifier", "admin", "notify", a["id"], rec)
        except Exception as e:  # noqa: BLE001 - any channel failure is retried
            self.fail_count += 1
            rec = {"alert": a["id"], "channel": ch_name, "level": job["level"], "ok": False,
                   "error": str(e)[:200], "attempt": job["attempt"], "t": time.time()}
            log.warning("notification via %s failed (attempt %d): %s", ch_name, job["attempt"] + 1, e)
            if job["attempt"] < self.max_retries:
                job = dict(job, attempt=job["attempt"] + 1)
                self._push(time.time() + self.backoff_s * 2 ** job["attempt"], job)
            else:
                self.audit("notifier", "admin", "notify_failed", a["id"], rec)
        self.sent.append(rec)
        del self.sent[:-500]
