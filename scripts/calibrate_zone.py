"""Installation-time calibration of a camera's false-alarm budget.

Record (or point at) footage of the zone in NORMAL operation - a few hours
across different times of day is ideal - and run:

    python scripts/calibrate_zone.py --zone "Corridor B" --video normal_day1.mp4 normal_day2.mp4 --alpha 0.01

For each hazard the smoothed risk score is summarised as the maximum over
non-overlapping windows (default 10 s).  The split-conformal threshold for
``alpha`` is then the alert line for that camera: on future normal footage
exchangeable with the calibration footage, at most ``alpha`` of windows will
raise an alarm (Vovk et al.; Angelopoulos & Bates 2021).  The result is
written to ``calibration/<zone>.json`` and loaded automatically by the copilot;
operator dismissals keep adding to it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgecopilot.io.video import VideoSource  # noqa: E402
from edgecopilot.pipeline import Copilot, load_config  # noqa: E402
from edgecopilot.xai.calibration import ConformalGate  # noqa: E402


def window_maxima(cop: Copilot, video: str, window_s: float, stride: int):
    cop.reset()
    maxima = {hz: [] for hz in cop.models}
    cur = {hz: -math.inf for hz in cop.models}
    w_start = None
    for frame, t in VideoSource(video, stride=stride):
        cop.process(frame, t)
        if w_start is None:
            w_start = t
        for key, st in cop.hyst.state.items():
            hz = key.split(":")[0]
            if hz in cur:
                cur[hz] = max(cur[hz], st["ema"])
        if t - w_start >= window_s:
            for hz in cur:
                if cur[hz] > -math.inf:
                    maxima[hz].append(cur[hz])
                cur[hz] = -math.inf
            w_start = t
    return maxima


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True)
    ap.add_argument("--video", nargs="+", required=True, help="footage of NORMAL operation only")
    ap.add_argument("--config", default="configs/campus.yaml")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cop = Copilot(cfg, root=root, zone=args.zone)
    scores = {hz: [] for hz in cop.models}
    for v in args.video:
        m = window_maxima(cop, v, args.window, args.stride)
        for hz in scores:
            scores[hz] += m[hz]
        print(f"{v}: " + ", ".join(f"{hz} {len(m[hz])} windows" for hz in m))
    out = {}
    for hz, s in scores.items():
        alpha = cfg["hazards"][hz].get("alpha") or args.alpha
        g = ConformalGate(alpha, scores=s)
        out[hz] = g.to_dict()
        tau = g.tau
        print(f"{hz:10s} n={len(s):5d} alpha={alpha}  " +
              (f"alert line = logit {tau:.2f} (p={1 / (1 + math.exp(-tau)):.3f})" if tau is not None
               else f"not enough windows (need >= {g.n_required}); keeping the configured threshold"))
    os.makedirs(os.path.dirname(cop.calib_path), exist_ok=True)
    with open(cop.calib_path, "w") as f:
        json.dump(out, f)
    print("wrote", cop.calib_path)


if __name__ == "__main__":
    main()
