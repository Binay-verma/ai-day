"""Where people leave the view, and where they are heading (no identity: image geometry only).

Two primitives the site-level features build on:

* **Exit events.** When a track ends, it is classified as having left through an image edge (its
  last box touched that edge, or was within ``margin`` of it and moving towards it) or as having
  vanished inside the image (occlusion, walked away into the depth, detector loss).
* **Heading.** For a moving track, the edge its current velocity points at (ray from the box
  centre to the image border) and the time to reach it.

Each camera's config maps edges to places (``cameras[].edges: {left: "West stairwell", ...}``), so
an edge becomes "the West stairwell" or "Lab Block A". Nothing here recognises a person: a track is
an anonymous box that lives for a few seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .tracker import Track

EDGES = ("left", "right", "top", "bottom")


@dataclass
class ExitEvent:
    track_id: int
    edge: Optional[str]          # None = vanished inside the image
    t: float                     # time the track was last seen
    box: np.ndarray
    velocity: np.ndarray         # px/s at the end
    low: bool = False            # was in a low posture (going down / lying) when last seen
    fall_prob: float = 0.0       # last fall score of that track
    fall_cues: List[dict] = field(default_factory=list)


@dataclass
class Heading:
    track_id: int
    edge: str
    eta_s: float
    speed_bh: float              # body-heights per second


def edge_of_box(box: np.ndarray, shape: Tuple[int, int], margin: float = 0.04) -> List[str]:
    """Edges the box touches (within ``margin`` of the image size)."""
    h, w = shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    out = []
    if x1 <= margin * w:
        out.append("left")
    if x2 >= (1 - margin) * w:
        out.append("right")
    if y1 <= margin * h:
        out.append("top")
    if y2 >= (1 - margin) * h:
        out.append("bottom")
    return out


def ray_to_border(c: np.ndarray, v: np.ndarray, shape: Tuple[int, int]) -> Tuple[Optional[str], float]:
    """Edge hit by the ray c + s*v (s > 0), and the time s to hit it."""
    h, w = shape[:2]
    best: Tuple[Optional[str], float] = (None, float("inf"))
    for edge, axis, bound in (("left", 0, 0.0), ("right", 0, float(w)), ("top", 1, 0.0), ("bottom", 1, float(h))):
        if abs(v[axis]) < 1e-9:
            continue
        s = (bound - c[axis]) / v[axis]
        if s <= 0:
            continue
        other = c[1 - axis] + s * v[1 - axis]
        lim = h if axis == 0 else w
        if -1e-6 <= other <= lim + 1e-6 and s < best[1]:
            best = (edge, float(s))
    return best


class ExitMonitor:
    """Per camera. Call :meth:`update` once per processed frame with the live tracks."""

    def __init__(self, margin: float = 0.06, toward_margin: float = 0.15, min_speed_bh: float = 0.25,
                 vel_window_s: float = 0.5):
        self.margin, self.toward_margin = margin, toward_margin
        self.min_speed_bh, self.vel_window_s = min_speed_bh, vel_window_s
        self._last: Dict[int, dict] = {}

    def reset(self) -> None:
        """Forget every track without reporting exits (scene change, camera moved, gap in the frames):
        people vanishing because the picture changed did not leave anywhere."""
        self._last.clear()

    def _velocity(self, tr: Track) -> np.ndarray:
        return np.asarray(tr.velocity(self.vel_window_s), dtype=np.float64)

    def update(self, tracks: Dict[int, Track], shape: Tuple[int, int],
               fall_state: Optional[Dict[int, dict]] = None) -> List[ExitEvent]:
        """``tracks``: the tracker's live tracks; ``fall_state``: optional per-track
        ``{"low": bool, "prob": float, "cues": [...]}`` from the fall branch."""
        fall_state = fall_state or {}
        for tid, tr in tracks.items():
            if getattr(tr, "misses", 0) == 0:  # only remember confirmed observations, not coasting predictions
                fs = fall_state.get(tid, {})
                prev = self._last.get(tid, {})
                self._last[tid] = {"box": np.asarray(tr.last.box, dtype=np.float64).copy(), "t": float(tr.last.t),
                                   "v": self._velocity(tr), "shape": shape[:2],
                                   "low": bool(fs.get("low", prev.get("low", False))) if fs else prev.get("low", False),
                                   "prob": float(fs.get("prob", prev.get("prob", 0.0))) if fs else prev.get("prob", 0.0),
                                   "cues": fs.get("cues", prev.get("cues", [])) if fs else prev.get("cues", [])}
        events = []
        for tid in [t for t in self._last if t not in tracks]:
            last = self._last.pop(tid)
            events.append(self._classify(tid, last))
        return events

    def _classify(self, tid: int, last: dict) -> ExitEvent:
        box, v, shape = last["box"], last["v"], last["shape"]
        touched = edge_of_box(box, shape, self.margin)
        edge: Optional[str] = None
        if touched:
            # several touched (a corner): the one the person was moving towards
            edge = max(touched, key=lambda e: _toward(v, e)) if len(touched) > 1 else touched[0]
        else:
            near = edge_of_box(box, shape, self.toward_margin)
            moving = [e for e in near if _toward(v, e) > 0.5 * (np.linalg.norm(v) + 1e-9)]
            if moving:
                edge = max(moving, key=lambda e: _toward(v, e))
        return ExitEvent(tid, edge, last["t"], box, v, last["low"], last["prob"], last["cues"])

    def headings(self, tracks: Dict[int, Track], shape: Tuple[int, int]) -> List[Heading]:
        """Edge each moving person is heading for. The edge score blends the direction of motion
        with the distance to that edge (``predict_edge``): on held-out walking footage this predicts
        the actual exit edge 2 s ahead 88 % of the time, against 31 % for "the nearest edge"
        (benchmarks/eval_exits.py)."""
        out = []
        for tid, tr in tracks.items():
            if getattr(tr, "misses", 0) > 0 or len(tr.history) < 3:
                continue
            box = np.asarray(tr.last.box, dtype=np.float64)
            v = self._velocity(tr)
            edge, eta, speed_bh = predict_edge(box, v, shape)
            if speed_bh < self.min_speed_bh or edge is None:
                continue
            out.append(Heading(tid, edge, eta, speed_bh))
        return out


def predict_edge(box: np.ndarray, v: np.ndarray, shape: Tuple[int, int]) -> Tuple[Optional[str], float, float]:
    """(edge, eta_s, speed in body-heights/s). Score per edge = (unit motion towards it) x
    min(speed, 2)/2 - (distance to it / image size): fast walkers are judged by direction, slow or
    hesitating people by where they already are."""
    h, w = shape[:2]
    c = (box[:2] + box[2:]) / 2
    hpx = max(float(box[3] - box[1]), 1.0)
    speed = float(np.linalg.norm(v))
    speed_bh = speed / hpx
    vn = v / (speed + 1e-9)
    dist = {"left": c[0], "right": w - c[0], "top": c[1], "bottom": h - c[1]}
    score = {e: _toward(vn, e) * min(speed_bh, 2.0) / 2.0 - dist[e] / (w if e in ("left", "right") else h) for e in dist}
    edge = max(score, key=lambda e: score[e])
    toward = _toward(v, edge)
    eta = float(dist[edge] / toward) if toward > 1e-6 else float("inf")
    return edge, eta, speed_bh


def _toward(v: np.ndarray, edge: str) -> float:
    return {"left": -v[0], "right": v[0], "top": -v[1], "bottom": v[1]}[edge]


def flow_summary(headings: Iterable[Heading], edges_map: Dict[str, str]) -> Dict[str, dict]:
    """Group live headings by the place each edge leads to."""
    out: Dict[str, dict] = {}
    for hd in headings:
        place = edges_map.get(hd.edge)
        if not place:
            continue
        g = out.setdefault(place, {"people": 0, "etas": [], "edge": hd.edge})
        g["people"] += 1
        g["etas"].append(hd.eta_s)
    for g in out.values():
        med = float(np.median(g["etas"])) if g["etas"] else float("inf")
        # None = not approaching (median person not moving towards that edge); never Infinity, which is not JSON
        g["eta_median_s"] = med if np.isfinite(med) else None
        del g["etas"]
    return out
