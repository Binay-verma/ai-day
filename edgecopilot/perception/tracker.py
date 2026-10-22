"""Lightweight multi-object tracker (IoU + centre distance, Hungarian matching).

ByteTrack-style two-stage association without a Kalman filter: cheap enough
for a 2-core edge CPU and good enough for short-horizon kinematics (falls,
exit flow).  Each track keeps a bounded history of boxes / keypoints with
timestamps so downstream rules can compute velocities in seconds, not frames.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    bb = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / (aa + bb - inter + 1e-9)


@dataclass
class TrackState:
    t: float
    box: np.ndarray
    keypoints: Optional[np.ndarray]
    score: float


@dataclass
class Track:
    track_id: int
    history: Deque[TrackState] = field(default_factory=lambda: deque(maxlen=90))
    misses: int = 0
    hits: int = 0

    @property
    def last(self) -> TrackState:
        return self.history[-1]

    def velocity(self, window_s: float = 0.5) -> np.ndarray:
        """Centre velocity in px/s over the last ``window_s`` seconds."""
        if len(self.history) < 2:
            return np.zeros(2)
        t_end = self.history[-1].t
        old = self.history[0]
        for st in self.history:
            if t_end - st.t <= window_s:
                old = st
                break
        dt = t_end - old.t
        if dt <= 1e-6:
            return np.zeros(2)
        c_new = (self.last.box[:2] + self.last.box[2:]) / 2
        c_old = (old.box[:2] + old.box[2:]) / 2
        return (c_new - c_old) / dt


class Tracker:
    def __init__(self, iou_thr: float = 0.3, max_misses: int = 15, dist_frac: float = 0.6):
        self.iou_thr = iou_thr
        self.max_misses = max_misses
        self.dist_frac = dist_frac
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, dets: List[Detection], t: float) -> List[Track]:
        ids = list(self.tracks.keys())
        prev = np.array([self.tracks[i].last.box for i in ids]) if ids else np.zeros((0, 4))
        cur = np.array([d.box for d in dets]) if dets else np.zeros((0, 4))
        matched_det = set()
        if len(ids) and len(dets):
            iou = iou_matrix(prev, cur)
            # centre-distance term rescues fast motion (e.g. a person falling)
            pc = (prev[:, :2] + prev[:, 2:]) / 2
            cc = (cur[:, :2] + cur[:, 2:]) / 2
            diag = np.hypot(prev[:, 2] - prev[:, 0], prev[:, 3] - prev[:, 1])[:, None] + 1e-6
            dist = np.linalg.norm(pc[:, None] - cc[None], axis=2) / diag
            cost = (1 - iou) + 0.5 * np.clip(dist, 0, 2)
            r, c = linear_sum_assignment(cost)
            for ri, ci in zip(r, c):
                if iou[ri, ci] >= self.iou_thr or dist[ri, ci] <= self.dist_frac:
                    tr = self.tracks[ids[ri]]
                    d = dets[ci]
                    tr.history.append(TrackState(t, d.box.copy(), None if d.keypoints is None else d.keypoints.copy(), d.score))
                    tr.misses = 0
                    tr.hits += 1
                    d.track_id = tr.track_id
                    matched_det.add(ci)
        matched_tracks = {d.track_id for d in dets if d.track_id >= 0}
        for i in ids:
            if i not in matched_tracks:
                self.tracks[i].misses += 1
                if self.tracks[i].misses > self.max_misses:
                    del self.tracks[i]
        for ci, d in enumerate(dets):
            if ci in matched_det:
                continue
            tr = Track(self._next_id)
            self._next_id += 1
            tr.history.append(TrackState(t, d.box.copy(), None if d.keypoints is None else d.keypoints.copy(), d.score))
            tr.hits = 1
            d.track_id = tr.track_id
            self.tracks[tr.track_id] = tr
        return [tr for tr in self.tracks.values() if tr.misses == 0]
