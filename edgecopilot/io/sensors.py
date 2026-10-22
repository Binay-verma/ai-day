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
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple


class SensorHub:
    FIELDS = ("smoke_ppm", "temp_c", "sound_db")

    def __init__(self, stale_s: float = 10.0):
        self.stale_s = stale_s
        self.hist: Dict[str, Dict[str, Deque[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=600)))
        self._replay: List[dict] = []
        self._replay_idx = 0

    def push(self, msg: dict) -> None:
        zone = msg.get("zone", "default")
        t = float(msg["t"])
        for k in self.FIELDS:
            if k in msg and msg[k] is not None:
                self.hist[zone][k].append((t, float(msg[k])))

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
    def cues(self, zone: str, t: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        values: Dict[str, float] = {}
        rel: Dict[str, float] = {}
        z = self.hist.get(zone) or self.hist.get("default")
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
        out = {}
        for zone, fields in self.hist.items():
            out[zone] = {k: {"value": q[-1][1], "age_s": round(t - q[-1][0], 1), "online": t - q[-1][0] <= self.stale_s}
                         for k, q in fields.items() if q}
        return out


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
