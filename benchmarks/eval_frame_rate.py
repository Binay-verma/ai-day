"""Fall accuracy when frames are dropped (why the browser camera aims for 15 FPS).

Replays the full production pipeline (the same chain as eval_left_view.py: tracker -> fall cues -> evidence
gates -> hysteresis -> alerts, "went down and left the view" on) on the cached GMDCSA24 detections, keeping k of
every n frames. The cache is at the clips' own rate (131 clips ~15 FPS, 29 clips ~7.5 FPS), so 3/4 is ~11 FPS,
1/2 ~7.5 FPS and 1/3 ~5 FPS for most clips. Protocols as in eval_left_view.py ("clip"; "hold" = the last
frame repeated for 4 s, the person stays down).

    python benchmarks/eval_frame_rate.py                       # -> benchmarks/results/frame_rate_gmdcsa24.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_left_view import run, summary  # noqa: E402

from edgecopilot.pipeline import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/gmdcsa24_lowconf.pkl")
    ap.add_argument("--keep", default="1/1,3/4,1/2,1/3", help="comma list of k/n: keep k of every n frames")
    ap.add_argument("--out", default="benchmarks/results/frame_rate_gmdcsa24.json")
    a = ap.parse_args()
    items = pickle.load(open(a.cache, "rb"))
    cfg = load_config("configs/campus.yaml")
    res = {"cache": a.cache, "config": "configs/campus.yaml (production)", "rows": {}}
    for keep in a.keep.split(","):
        k, n = map(int, keep.split("/"))
        its = [dict(it, frames=[f for i, f in enumerate(it["frames"]) if i % n < k]) for it in items]
        for proto in ("clip", "hold"):
            s = summary(run(its, cfg, True, proto))
            res["rows"][f"{keep} {proto}"] = s
            print(keep, proto, s, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
