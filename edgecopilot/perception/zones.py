"""Restricted-area cues (intrusion / after-hours presence), interpretable by construction.

A region is a polygon in normalised image coordinates (0..1) with an optional time window
during which presence is *not* permitted, e.g. a chemistry lab after 20:00 or a rooftop at
any time. For every region the extractor reports, per frame:

* ``dwell_time``  - seconds the region has been continuously occupied (gaps < 2 s bridged)
* ``after_hours`` - 1 when the local time is inside the region's forbidden window, else 0
* ``people``      - number of tracked people whose foot point is inside the region

The additive "intrusion" model in configs/campus.yaml fuses them like every other hazard,
so the alert comes with the same exact ledger, counterfactual ("would not have fired if
the stay had been under 12 s") and verified explanation.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _parse_window(hours: Optional[str]) -> Optional[Tuple[int, int]]:
    """'20:00-06:00' -> (1200, 360) minutes; None = always restricted."""
    if not hours:
        return None
    a, b = hours.split("-")
    to_min = lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1])  # noqa: E731
    return to_min(a.strip()), to_min(b.strip())


def in_window(window: Optional[Tuple[int, int]], minute_of_day: int) -> bool:
    if window is None:
        return True
    a, b = window
    return a <= minute_of_day < b if a < b else (minute_of_day >= a or minute_of_day < b)


@dataclass
class Region:
    name: str
    polygon: np.ndarray  # (N, 2) normalised
    window: Optional[Tuple[int, int]] = None  # forbidden hours; None = always forbidden
    occupied_since: Optional[float] = None
    last_seen: Optional[float] = None
    tracks_inside: List[int] = field(default_factory=list)


class ZoneRules:
    def __init__(self, regions: Sequence[dict], gap_s: float = 2.0):
        self.regions = [Region(r["name"], np.asarray(r["polygon"], np.float32), _parse_window(r.get("hours")))
                        for r in regions]
        self.gap_s = gap_s

    def reset(self) -> None:
        for r in self.regions:
            r.occupied_since = r.last_seen = None
            r.tracks_inside = []

    def __call__(self, people: Sequence, frame_shape: Tuple[int, int], t: float,
                 wall_t: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        h, w = frame_shape[:2]
        now = _dt.datetime.fromtimestamp(wall_t) if wall_t is not None and wall_t > 1e9 else _dt.datetime.now()
        minute = now.hour * 60 + now.minute
        out: Dict[str, Dict[str, float]] = {}
        for r in self.regions:
            poly = (r.polygon * np.array([w, h], np.float32)).reshape(-1, 1, 2)
            inside = []
            for d in people:
                fx, fy = float((d.box[0] + d.box[2]) / 2), float(d.box[3])  # foot point
                if cv2.pointPolygonTest(poly, (fx, fy), False) >= 0:
                    inside.append(getattr(d, "track_id", -1))
            if inside:
                if r.occupied_since is None or (r.last_seen is not None and t - r.last_seen > self.gap_s):
                    r.occupied_since = t
                r.last_seen = t
            elif r.last_seen is not None and t - r.last_seen > self.gap_s:
                r.occupied_since = None
            r.tracks_inside = inside
            dwell = (t - r.occupied_since) if r.occupied_since is not None else 0.0
            # "outside permitted hours" is evidence only while someone is actually inside
            out[r.name] = {"dwell_time": float(dwell), "after_hours": 1.0 if (inside and in_window(r.window, minute)) else 0.0,
                           "people": float(len(inside))}
        return out
