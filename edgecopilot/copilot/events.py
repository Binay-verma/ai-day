"""Alert objects, the append-only event log and operator feedback records."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class Alert:
    hazard: str
    subject: str
    zone: str
    t_video: float  # stream time of onset (s)
    prob: float
    severity: str
    explanation: dict
    decision: dict  # counterfactuals, sufficient reason, odds multipliers
    verification: dict  # grounding check of the explanation text
    recommendations: List[dict]
    cues: List[dict]
    triage: dict = field(default_factory=dict)  # ALARM | VERIFY with reasons
    snapshot_jpg_b64: str = ""
    evidence_jpg_b64: str = ""
    processing_ms: float = 0.0  # compute time for the frame that raised it
    explain_ms: float = 0.0  # time to build saliency + explanation
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    wall_time: float = field(default_factory=time.time)
    status: str = "open"  # open | confirmed | dismissed | resolved
    operator_note: str = ""
    peak_prob: float = 0.0
    peak_ema: float = -1e9  # peak smoothed logit while active (calibration units)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("snapshot_jpg_b64", None)
        d.pop("evidence_jpg_b64", None)
        return d


class EventLog:
    """Thread-safe in-memory store mirrored to a JSONL file (no images on disk)."""

    def __init__(self, path: Optional[str] = None, max_items: int = 500):
        self.path = path
        self.max_items = max_items
        self.alerts: Dict[str, Alert] = {}
        self.order: List[str] = []
        self.lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _write(self, rec: dict) -> None:
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")

    def add(self, alert: Alert) -> Alert:
        with self.lock:
            alert.peak_prob = max(alert.peak_prob, alert.prob)
            self.alerts[alert.id] = alert
            self.order.append(alert.id)
            if len(self.order) > self.max_items:
                old = self.order.pop(0)
                self.alerts.pop(old, None)
        self._write({"type": "alert", **alert.summary()})
        return alert

    def update_status(self, alert_id: str, status: str, note: str = "") -> Optional[Alert]:
        with self.lock:
            a = self.alerts.get(alert_id)
            if a is None:
                return None
            a.status = status
            a.operator_note = note
        self._write({"type": "feedback", "id": alert_id, "status": status, "note": note, "wall_time": time.time()})
        return a

    def recent(self, n: int = 50) -> List[Alert]:
        with self.lock:
            return [self.alerts[i] for i in self.order[-n:]][::-1]
