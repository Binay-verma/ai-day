"""Crowd cues: occupancy, motion energy and coordinated (exit / panic) movement.

Dense Farneback optical flow is computed on a small grey image (default
width 320) and summarised into interpretable numbers:

* ``occupancy``      - people counted / zone capacity
* ``speed_ratio``    - speed of the fastest 10 % of moving pixels (body-heights/s)
                       divided by the scene's adaptive "normal" (running median)
* ``motion_area``    - fraction of the frame that is moving
* ``coherence``      - |mean unit flow vector| over moving pixels: 1 = everyone
                       moving the same way (rapid exit), 0 = random
* ``divergence``     - radial outward flow relative to the motion centroid
                       (sudden dispersal, as in the UMN panic scenes)
* ``ego_motion``     - median flow of the whole frame: > 0 when the camera itself
                       moves; scene-baseline cues are then down-weighted
* ``acceleration``   - rise of speed over the last second, normalised
* ``surge``          - peak speed ratio over the last 2 s (a run is still
                       explainable after people have stopped or left)
* ``exodus``         - share of the people seen in the last 3 s who have left

A flow-magnitude heat-map is kept as the visual evidence (saliency) for crowd
alerts, so the operator sees *where* the abnormal motion is.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np


class CrowdFeatureExtractor:
    def __init__(self, width: int = 320, baseline_s: float = 20.0, min_mag_px: float = 0.6,
                 exit_direction: Optional[Tuple[float, float]] = None,
                 ignore_regions: Optional[list] = None, surge_s: float = 2.0, exodus_s: float = 3.0):
        self.width = width
        # normalised [x1, y1, x2, y2] rectangles to ignore (burnt-in timestamps, banners, screens)
        self.ignore_regions = ignore_regions or []
        self.baseline_s = baseline_s
        self.min_mag_px = min_mag_px
        self.exit_direction = None if exit_direction is None else np.asarray(exit_direction, float)
        self.prev: Optional[np.ndarray] = None
        self.prev_t: Optional[float] = None
        self.speed_hist: Deque[Tuple[float, float]] = deque(maxlen=4000)
        self.recent: Deque[Tuple[float, float]] = deque(maxlen=200)
        self.heatmap: Optional[np.ndarray] = None
        self.last_values: Dict[str, float] = {}
        self.ph_ema: Optional[float] = None
        self.surge_s, self.exodus_s = surge_s, exodus_s
        self.ratio_hist: Deque[Tuple[float, float]] = deque()
        self.count_hist: Deque[Tuple[float, float]] = deque()

    def idle(self, t: float, n_people: int) -> Dict[str, float]:
        """Cheap path when nobody is in view: keep the head-count history, skip optical flow."""
        self.count_hist.append((t, n_people))
        while self.count_hist and t - self.count_hist[0][0] > self.exodus_s:
            self.count_hist.popleft()
        self.prev, self.prev_t = None, None  # next flow pair starts fresh (no bogus dt)
        self.heatmap = None
        self.last_values = {"occupancy": 0.0, "people": float(n_people)}
        return self.last_values

    def reset(self) -> None:
        self.prev = None
        self.prev_t = None
        self.speed_hist.clear()
        self.recent.clear()
        self.heatmap = None
        self.ph_ema = None
        self.ratio_hist.clear()
        self.count_hist.clear()

    def _baseline(self, t: float) -> float:
        vals = [s for (ts, s) in self.speed_hist if t - ts <= self.baseline_s]
        if len(vals) < 10:
            return float("nan")
        return float(np.median(vals))

    def __call__(self, frame_bgr: np.ndarray, t: float, n_people: int = 0, capacity: float = 50.0,
                 person_height_px: Optional[float] = None, alert_active: bool = False) -> Dict[str, float]:
        h0, w0 = frame_bgr.shape[:2]
        scale = self.width / w0
        small = cv2.resize(frame_bgr, (self.width, int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        values: Dict[str, float] = {"occupancy": n_people / max(capacity, 1.0), "people": float(n_people)}
        if self.prev is None or self.prev.shape != gray.shape:
            self.prev, self.prev_t = gray, t
            self.last_values = values
            return values
        dt = t - self.prev_t if self.prev_t is not None else 0.0
        if not 1e-3 < dt < 2.0:
            dt = 1 / 25.0  # first frame after a gap / clock jump: assume one nominal frame interval
        flow = cv2.calcOpticalFlowFarneback(self.prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev, self.prev_t = gray, t
        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        moving = mag > self.min_mag_px * max(dt * 25, 0.2)  # threshold scaled to a 25 fps reference
        H, W = mag.shape
        for x1, y1, x2, y2 in self.ignore_regions:
            moving[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = False
            mag[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = 0
        frac = float(moving.mean())
        # camera ego-motion: in a static camera most pixels do not move, so the
        # median flow is ~0; a panning / hand-held / robot camera moves everything
        ego = float(np.median(mag)) / dt / mag.shape[0]  # frame-heights per second
        # normalise speed by a *stable* person scale (EMA of median person height, in the
        # small image) so units are body-heights / s and do not jump when detections drop out
        if person_height_px:
            ph = person_height_px * scale
            self.ph_ema = ph if self.ph_ema is None else 0.95 * self.ph_ema + 0.05 * ph
        norm = self.ph_ema if self.ph_ema else gray.shape[0] / 4.0
        fast = 0.0
        if moving.sum() >= 20:
            speed = float(mag[moving].mean()) / dt / max(norm, 1.0)
            fast = float(np.percentile(mag[moving], 90)) / dt / max(norm, 1.0)
            ux, uy = fx[moving] / (mag[moving] + 1e-6), fy[moving] / (mag[moving] + 1e-6)
            coherence = float(np.hypot(ux.mean(), uy.mean()))
            ys, xs = np.nonzero(moving)
            cx, cy = xs.mean(), ys.mean()
            rx, ry = xs - cx, ys - cy
            rn = np.hypot(rx, ry) + 1e-6
            divergence = float(np.mean((fx[moving] * rx + fy[moving] * ry) / rn) / dt / max(norm, 1.0))
            exit_align = float("nan")
            if self.exit_direction is not None:
                d = self.exit_direction / (np.linalg.norm(self.exit_direction) + 1e-9)
                exit_align = float(np.mean(ux * d[0] + uy * d[1]))
        else:
            speed, coherence, divergence, exit_align = 0.0, 0.0, 0.0, float("nan")
        # energy = speed x area; "fast" = speed of the fastest 10 % of moving pixels
        # (running people), compared with this scene's own adaptive normal
        energy = speed * frac
        base = self._baseline(t)
        if not alert_active and fast > 0:  # do not let an emergency become the new normal
            self.speed_hist.append((t, fast))
        ratio = fast / base if base == base and base > 1e-4 else 1.0
        self.recent.append((t, fast))
        past = [e for (ts, e) in self.recent if 0.8 <= t - ts <= 1.2]
        accel = (fast - past[0]) / (base if base == base and base > 1e-4 else max(fast, 1e-3)) if past else 0.0
        # short memories that make transient events explainable a moment later:
        # surge = peak speed ratio over the last surge_s; exodus = share of the
        # people present in the last exodus_s who have left the view
        clipped = float(np.clip(ratio, 0, 50))
        self.ratio_hist.append((t, clipped))
        while self.ratio_hist and t - self.ratio_hist[0][0] > self.surge_s:
            self.ratio_hist.popleft()
        surge = max(r for _, r in self.ratio_hist)
        # de-flicker the head-count (median over 0.5 s) before measuring departures
        self.count_hist.append((t, n_people))
        while self.count_hist and t - self.count_hist[0][0] > self.exodus_s:
            self.count_hist.popleft()
        smooth = [(ts, float(np.median([c for t2, c in self.count_hist if 0 <= ts - t2 <= 0.5])))
                  for ts, _ in self.count_hist]
        now = smooth[-1][1]
        pmax = max(c for _, c in smooth)
        exodus = (pmax - now) / pmax if pmax >= 3 else 0.0
        heat = cv2.GaussianBlur(mag, (0, 0), 3)
        self.heatmap = heat / (heat.max() + 1e-6)
        values.update({
            "speed": speed,
            "ego_motion": ego,
            "fast_speed": fast,
            "energy": energy,
            "speed_ratio": clipped,
            "surge": float(surge),
            "exodus": float(exodus),
            "motion_area": frac,
            "coherence": coherence,
            "divergence": divergence,
            "acceleration": float(np.clip(accel, -50, 50)),
        })
        if exit_align == exit_align:
            values["exit_alignment"] = exit_align
        self.last_values = values
        return values
