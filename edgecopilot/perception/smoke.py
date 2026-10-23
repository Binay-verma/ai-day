"""Physically interpretable smoke / fire image cues (complement the detector).

The learned detector says *what* it saw; these cues say *how the scene
changed*, in units an operator can reason about:

* ``smoke_density``  - rise of the dark channel (haze) relative to the
                       scene's own adaptive baseline, max over a 4x4 grid of
                       cells (smoke is local first).  0 = clear, ~0.3+ = thick.
* ``contrast_loss``  - fractional drop of local contrast in that cell vs its
                       baseline (smoke washes out edges).
* ``smoke_growth``   - rate of change of smoke_density per second.
* ``fire_pixels``    - fraction (per mille) of pixels that are flame-coloured
                       (R > G > B, bright, saturated) *and* flickering.

Dark channel prior (He et al., CVPR 2009): in haze-free outdoor/indoor
patches, at least one colour channel is near zero; smoke/haze lifts it.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np


class SmokeCueExtractor:
    def __init__(self, width: int = 160, grid: int = 4, patch: int = 7, tau_up_s: float = 900.0,
                 tau_down_s: float = 20.0, warmup_s: float = 1.0):
        self.width = width
        self.grid = grid
        self.patch = patch
        # asymmetric, time-based background: the haze baseline may *rise* only very slowly
        # (tau 15 min) so slowly building smoke is not absorbed into "normal", but it falls
        # back quickly (tau 20 s) when the scene clears or gets darker
        self.tau_up_s, self.tau_down_s, self.warmup_s = tau_up_s, tau_down_s, warmup_s
        self.t0: Optional[float] = None
        self.prev_t: Optional[float] = None
        self.base_dark: Optional[np.ndarray] = None
        self.base_contrast: Optional[np.ndarray] = None
        self.prev_fire: Optional[np.ndarray] = None
        self.hist: Deque[Tuple[float, float]] = deque(maxlen=200)
        self.n = 0
        self.dark_map: Optional[np.ndarray] = None
        self.cell_delta: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.__init__(self.width, self.grid, self.patch, self.tau_up_s, self.tau_down_s, self.warmup_s)  # type: ignore[misc]

    def _cells(self, img: np.ndarray, fn) -> np.ndarray:
        h, w = img.shape[:2]
        g = self.grid
        out = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                out[i, j] = fn(img[i * h // g:(i + 1) * h // g, j * w // g:(j + 1) * w // g])
        return out

    def __call__(self, frame_bgr: np.ndarray, t: float, alert_active: bool = False) -> Dict[str, float]:
        h0, w0 = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (self.width, max(8, int(round(h0 * self.width / w0)))), interpolation=cv2.INTER_AREA)
        f = small.astype(np.float32) / 255.0
        dark = cv2.erode(f.min(axis=2), np.ones((self.patch, self.patch), np.uint8))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        d_cells = self._cells(dark, np.mean)
        c_cells = self._cells(lap, np.mean) + 1e-4
        self.dark_map = dark
        if self.base_dark is None:
            self.base_dark, self.base_contrast = d_cells.copy(), c_cells.copy()
            self.t0 = t
        dt = 0.0 if self.prev_t is None else max(0.0, min(t - self.prev_t, 5.0))
        self.prev_t = t
        in_warmup = (t - (self.t0 if self.t0 is not None else t)) < self.warmup_s
        warm = 0.0 if in_warmup else 1.0  # the first second of video is the reference
        delta = (d_cells - self.base_dark) * warm
        closs = np.clip(1 - c_cells / self.base_contrast, -1, 1) * warm
        # combined per-cell haze score: brighter dark channel AND fewer edges
        score = np.clip(delta, 0, 1) * (0.5 + 0.5 * np.clip(closs, 0, 1))
        k = np.unravel_index(np.argmax(score), score.shape)
        self.cell_delta = score
        density = float(np.clip(delta[k], 0, 1))
        contrast_loss = float(np.clip(closs[k], 0, 1))
        if in_warmup:
            self.base_dark = 0.7 * self.base_dark + 0.3 * d_cells  # type: ignore[operator]
            self.base_contrast = 0.7 * self.base_contrast + 0.3 * c_cells  # type: ignore[operator]
        elif not alert_active and dt > 0:
            up = d_cells > self.base_dark  # haze rising -> follow very slowly
            a_dark = np.where(up, 1 - np.exp(-dt / self.tau_up_s), 1 - np.exp(-dt / self.tau_down_s))
            self.base_dark = (1 - a_dark) * self.base_dark + a_dark * d_cells  # type: ignore[operator]
            loss = c_cells < self.base_contrast  # edges disappearing -> follow very slowly
            a_c = np.where(loss, 1 - np.exp(-dt / self.tau_up_s), 1 - np.exp(-dt / self.tau_down_s))
            self.base_contrast = (1 - a_c) * self.base_contrast + a_c * c_cells  # type: ignore[operator]
        self.n += 1
        # growth: density change per second over ~1 s
        self.hist.append((t, density))
        past = [(ts, d) for (ts, d) in self.hist if 0.8 <= t - ts <= 1.5]
        growth = (density - past[0][1]) / max(t - past[0][0], 1e-3) if past else 0.0

        # flame colour + flicker
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        b, g_, r = small[..., 0].astype(int), small[..., 1].astype(int), small[..., 2].astype(int)
        fire = (r > 190) & (r > g_ + 20) & (g_ > b) & (hsv[..., 1] > 80) & (hsv[..., 2] > 180)
        changed = 0.0
        if self.prev_fire is not None and self.prev_fire.shape == fire.shape:
            changed = float((fire ^ self.prev_fire).mean())  # flames flicker, lamps don't
        self.prev_fire = fire
        fire_frac = float(fire.mean())
        return {
            "smoke_density": density,
            "contrast_loss": contrast_loss,
            "smoke_growth": float(growth),
            "fire_pixels": 1000.0 * fire_frac,
            "fire_flicker": 1000.0 * float(changed),
        }
