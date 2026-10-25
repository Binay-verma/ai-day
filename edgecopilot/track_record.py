"""Operator track record (idea ported from the team's Argus project, re-implemented here).

For each hazard on each camera: of the alerts an operator reviewed in the last ``window_days``,
how many were confirmed as real and how many dismissed as false alarms, with a Wilson 95 % score
interval. The console shows it next to every alert ("operators confirmed 9 of 11 fall alerts on this
camera, 82 % [52-95 %]"), so trust in an alert is earned from this site's own record, not claimed.
Alerts still open, trouble signals and self-tests are not counted.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

NOT_COUNTED = ("camera_fault", "self_test")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


class TrackRecord:
    def __init__(self, store, window_days: float = 30.0, ttl_s: float = 30.0, min_reviews: int = 3):
        self.store, self.window_s, self.ttl_s, self.min_reviews = store, window_days * 86400, ttl_s, min_reviews
        self._cache: Optional[Dict[Tuple[str, str], list]] = None
        self._at = 0.0
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    def counts(self, now: Optional[float] = None) -> Dict[Tuple[str, str], list]:
        now = time.time() if now is None else now
        with self._lock:
            if self._cache is not None and now - self._at < self.ttl_s:
                return self._cache
        tally: Dict[Tuple[str, str], list] = {}
        for a in self.store.list_alerts(5000, since=now - self.window_s):
            st, hz = a.get("status"), a.get("hazard")
            if hz in NOT_COUNTED or st not in ("confirmed", "dismissed"):
                continue
            for key in ((hz, a.get("camera_id") or ""), (hz, "*")):
                t = tally.setdefault(key, [0, 0])
                t[0 if st == "confirmed" else 1] += 1
        with self._lock:
            self._cache, self._at = tally, now
        return tally

    def record_for(self, hazard: str, camera: str, now: Optional[float] = None) -> Dict[str, Any]:
        c = self.counts(now)
        conf, dism = c.get((hazard, camera), [0, 0])
        scope = "camera"
        if conf + dism < self.min_reviews:  # too few on this camera: fall back to the site-wide record
            conf, dism = c.get((hazard, "*"), [0, 0])
            scope = "site"
        n = conf + dism
        lo, hi = wilson(conf, n)
        days = round(self.window_s / 86400)
        if n == 0:
            text = f"No reviewed {hazard.replace('_', ' ')} alerts in the last {days} days yet."
        else:
            where = "on this camera" if scope == "camera" else "across the site"
            text = (f"Operators confirmed {conf} of {n} reviewed {hazard.replace('_', ' ')} alerts {where} in the last "
                    f"{days} days ({conf / n:.0%}, 95 % CI {lo:.0%}-{hi:.0%}).")
        return {"hazard": hazard, "camera": camera, "scope": scope, "confirmed": conf, "dismissed": dism, "reviewed": n,
                "precision": (conf / n) if n else None, "ci95": [round(lo, 3), round(hi, 3)], "window_days": days,
                "text": text}

    def table(self, now: Optional[float] = None) -> list:
        rows = []
        for (hz, cam), (conf, dism) in sorted(self.counts(now).items()):
            n = conf + dism
            lo, hi = wilson(conf, n)
            rows.append({"hazard": hz, "camera": cam, "confirmed": conf, "dismissed": dism, "reviewed": n,
                         "precision": conf / n if n else None, "ci95": [round(lo, 3), round(hi, 3)]})
        return rows
