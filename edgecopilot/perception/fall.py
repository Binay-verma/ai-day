"""Pose-kinematics fall cues (interpretable by construction).

From COCO-17 key-points of a tracked person we compute physically meaningful
cues that the additive fall model fuses:

* ``torso_angle``   - angle of the hip->shoulder vector from vertical (deg)
* ``descent_speed`` - peak downward speed of the hip centre over the last few
                      seconds, in body-heights per second (scale invariant)
* ``height_ratio``  - current box height / the person's standing height
                      (reported as ``height_drop`` = 1 - ratio)
* ``aspect_ratio``  - box width / height (lying people are wide)
* ``ground_time``   - seconds spent in a lying posture with little motion

Each cue comes with a reliability in [0, 1] from key-point confidence so an
occluded person yields *weaker* evidence instead of a confident wrong answer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .tracker import Track

L_SH, R_SH, L_HIP, R_HIP, NOSE = 5, 6, 11, 12, 0
KP_CONF = 0.3


@dataclass
class FallState:
    standing_heights: list = field(default_factory=list)
    lying_since: Optional[float] = None
    peak_descent: float = 0.0
    peak_t: float = -1e9
    hip_hist: list = field(default_factory=list)  # [(t, y)] for smoothed velocity


def _mid(kp: np.ndarray, a: int, b: int) -> Optional[np.ndarray]:
    pts = [kp[i, :2] for i in (a, b) if kp[i, 2] >= KP_CONF]
    if not pts:
        return None
    return np.mean(pts, axis=0)


def torso_angle(kp: np.ndarray) -> Optional[float]:
    sh, hip = _mid(kp, L_SH, R_SH), _mid(kp, L_HIP, R_HIP)
    if sh is None or hip is None:
        return None
    v = sh - hip
    if np.linalg.norm(v) < 1e-3:
        return None
    # angle from "up" (0,-1): 0 = upright, 90 = horizontal, >90 = head below hips
    return float(math.degrees(math.atan2(abs(v[0]), -v[1])))


class FallFeatureExtractor:
    """Keeps per-track state and produces fall cue values each frame."""

    def __init__(self, descent_window_s: float = 3.0, lying_angle: float = 55.0,
                 still_speed: float = 0.25, vel_window_s: float = 0.3):
        self.descent_window_s = descent_window_s
        self.vel_window_s = vel_window_s
        self.lying_angle = lying_angle
        self.still_speed = still_speed
        self.state: Dict[int, FallState] = {}

    def reset(self) -> None:
        self.state.clear()

    def standing_height(self, st: FallState, cur_h: float) -> float:
        if st.standing_heights:
            return float(np.percentile(st.standing_heights, 90))
        return cur_h

    def __call__(self, tr: Track, t: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        st = self.state.setdefault(tr.track_id, FallState())
        last = tr.last
        box = last.box
        w, h = float(box[2] - box[0]), float(box[3] - box[1])
        kp = last.keypoints
        ang = torso_angle(kp) if kp is not None else None
        if ang is None:  # fall back to box geometry when torso key-points are missing
            ang_box = 90.0 * min(1.0, max(0.0, (w / max(h, 1) - 0.4) / 1.2))
            ang, rel_ang = ang_box, 0.4
        else:
            rel_ang = float(np.mean([kp[i, 2] for i in (L_SH, R_SH, L_HIP, R_HIP)]))
        upright = ang < 30 and h > w
        if upright:
            st.standing_heights.append(h)
            st.standing_heights = st.standing_heights[-60:]
        H = max(self.standing_height(st, h), 1.0)

        # hip-centre (or box-centre) vertical velocity over ~vel_window_s, body-heights / s
        hip = _mid(kp, L_HIP, R_HIP) if kp is not None else None
        y = float(hip[1]) if hip is not None else float((box[1] + box[3]) / 2)
        st.hip_hist.append((t, y))
        st.hip_hist = [p for p in st.hip_hist if t - p[0] <= 1.0]
        prev = st.hip_hist[:-1]
        in_win = [p for p in prev if t - p[0] <= self.vel_window_s]
        ref = in_win[0] if in_win else (prev[-1] if prev else (t, y))
        vy = 0.0
        if t - ref[0] > 1e-3:
            vy = float(np.clip((y - ref[1]) / (t - ref[0]) / H, -5, 5))
        if t - st.peak_t > self.descent_window_s:
            st.peak_descent = 0.0
        if vy > st.peak_descent:
            st.peak_descent, st.peak_t = vy, t

        speed = float(np.linalg.norm(tr.velocity(0.5))) / H
        lying = ang >= self.lying_angle or (w > 1.1 * h and h < 0.7 * H)
        if lying and speed < self.still_speed:
            if st.lying_since is None:
                st.lying_since = t
        elif not lying:
            st.lying_since = None
        ground_time = 0.0 if st.lying_since is None else t - st.lying_since

        values = {
            "torso_angle": float(ang),
            "descent_speed": float(st.peak_descent),
            "height_drop": float(np.clip(1 - h / H, 0, 1)),
            "aspect_ratio": float(w / max(h, 1.0)),
            "ground_time": float(ground_time),
        }
        rel = {"torso_angle": rel_ang, "descent_speed": min(1.0, 0.5 + 0.5 * len(st.standing_heights) / 10),
               "height_drop": min(1.0, len(st.standing_heights) / 10)}
        return values, rel

    def prune(self, live_ids) -> None:
        for k in list(self.state):
            if k not in live_ids:
                del self.state[k]
