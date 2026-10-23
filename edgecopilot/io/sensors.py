"""Sensor hub: latest IoT readings per zone with health-aware reliability.

Readings arrive as small JSON messages (from the ESP32 node in
``hardware/esp32_sensor_node``, an HTTP POST, or a CSV replay):

    {"zone": "Lab Block A", "t": 12.3, "smoke_ppm": 410, "temp_c": 31.2, "sound_db": 78}

Derived cues: ``smoke_ppm``, ``temp_rise`` (deg C per minute over the last
60 s) and ``sound_db``.  A sensor that has not reported for ``stale_s``
seconds gets reliability 0, so the fusion model ignores it and the
explanation can say the sensor was offline instead of trusting old data.
"""
from __future__ import annotations

import csv
import math
import threading
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple


class SensorHub:
    """Thread-safe: HTTP/MQTT threads push while camera threads read.

    ``zones`` (optional) restricts which zone names are accepted, so a device
    cannot grow memory by inventing zones; without it at most ``max_zones``
    zones are kept.  ``fallback_zone`` is used only when a camera's own zone has
    no sensors (CSV replay / single-room demos); in a real deployment leave it
    ``None`` so a sensor in one room can never vouch for another room.
    """

    FIELDS = ("smoke_ppm", "temp_c", "sound_db")

    def __init__(self, stale_s: float = 10.0, zones: Optional[Iterable[str]] = None, max_zones: int = 64,
                 fallback_zone: Optional[str] = None):
        self.stale_s = stale_s
        self.allowed = set(zones) if zones is not None else None
        self.max_zones = max_zones
        self.fallback_zone = fallback_zone
        self.hist: Dict[str, Dict[str, Deque[Tuple[float, float]]]] = {}
        self._lock = threading.RLock()
        self._replay: List[dict] = []
        self._replay_idx = 0
        self.rejected = 0

    def push(self, msg: dict) -> bool:
        zone = str(msg.get("zone", "default"))[:60]
        try:
            t = float(msg["t"])
        except (KeyError, TypeError, ValueError):
            self.rejected += 1
            return False
        if not math.isfinite(t):
            self.rejected += 1
            return False
        with self._lock:
            if zone not in self.hist:
                if (self.allowed is not None and zone not in self.allowed) or len(self.hist) >= self.max_zones:
                    self.rejected += 1
                    return False
                self.hist[zone] = {}
            z = self.hist[zone]
            for k in self.FIELDS:
                v = msg.get(k)
                if v is None:
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    z.setdefault(k, deque(maxlen=600)).append((t, v))
        return True

    # ------------------------------------------------------------- replay
    def load_csv(self, path: str) -> None:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["t"] = float(r["t"])
            for k in self.FIELDS:
                if r.get(k) not in (None, ""):
                    r[k] = float(r[k])
                else:
                    r.pop(k, None)
        self._replay = sorted(rows, key=lambda r: r["t"])
        self._replay_idx = 0

    def advance(self, t: float) -> None:
        """Feed replayed readings with timestamp <= t."""
        while self._replay_idx < len(self._replay) and self._replay[self._replay_idx]["t"] <= t:
            self.push(self._replay[self._replay_idx])
            self._replay_idx += 1

    # ------------------------------------------------------------- cues
    def _snapshot(self, zone: str) -> Dict[str, List[Tuple[float, float]]]:
        with self._lock:
            z = self.hist.get(zone)
            if not z and self.fallback_zone is not None:
                z = self.hist.get(self.fallback_zone)
            return {k: list(q) for k, q in (z or {}).items()}

    def cues(self, zone: str, t: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        values: Dict[str, float] = {}
        rel: Dict[str, float] = {}
        z = self._snapshot(zone)
        if not z:
            return values, rel
        for k, out in (("smoke_ppm", "smoke_ppm"), ("sound_db", "sound_db")):
            q = z.get(k)
            if q:
                ts, v = q[-1]
                values[out] = v
                rel[out] = 1.0 if t - ts <= self.stale_s else 0.0
        q = z.get("temp_c")
        if q and len(q) >= 2:
            ts, v = q[-1]
            old = next(((a, b) for (a, b) in q if ts - a <= 60.0), q[0])
            if ts - old[0] >= 5.0:
                values["temp_rise"] = (v - old[1]) / (ts - old[0]) * 60.0
                rel["temp_rise"] = 1.0 if t - ts <= self.stale_s else 0.0
        return values, rel

    def status(self, t: float) -> Dict[str, dict]:
        with self._lock:
            snap = {zone: {k: q[-1] for k, q in fields.items() if q} for zone, fields in self.hist.items()}
        return {zone: {k: {"value": v, "age_s": round(t - ts, 1), "online": t - ts <= self.stale_s}
                       for k, (ts, v) in fields.items()} for zone, fields in snap.items()}


def synthetic_fire_trace(zone: str, t0: float, t1: float, fire_at: Optional[float], dt: float = 1.0) -> List[dict]:
    """Deterministic MQ-2 / DHT22 / sound trace for demos and benchmarks."""
    import math
    rows = []
    t = t0
    while t <= t1:
        base_ppm = 120 + 8 * math.sin(t / 7.0)
        temp = 27.0 + 0.2 * math.sin(t / 30.0)
        snd = 55 + 3 * math.sin(t / 3.0)
        if fire_at is not None and t >= fire_at:
            k = t - fire_at
            base_ppm += min(900.0, 60.0 * k)
            temp += min(12.0, 0.08 * k)
            snd += min(20.0, 2.0 * k)
        rows.append({"zone": zone, "t": round(t, 2), "smoke_ppm": round(base_ppm, 1),
                     "temp_c": round(temp, 2), "sound_db": round(snd, 1)})
        t += dt
    return rows
