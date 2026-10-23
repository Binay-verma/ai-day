"""Hard-negative mining for the fire/smoke detector.

Takes a folder of images that contain no fire or smoke (e.g. the cair "Normal Images"
set), drops every image within ``--dup-bits`` perceptual-hash bits of ANY evaluation
image in data/fire_splits.json (so evaluation stays leakage-free), runs the current
detector, and writes the confounders it fires on (score >= ``--min-score``) plus a
random share of clean ones into the YOLO training split as empty-label images.

    python scripts/mine_hard_negatives.py --src /path/to/normal_images --model models/fire_smoke.onnx
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_fire_data import min_dist, phash  # noqa: E402

from edgecopilot.perception.detector import OnnxYolo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--model", default="models/fire_smoke.onnx")
    ap.add_argument("--manifest", default="data/fire_splits.json")
    ap.add_argument("--out", default="data/fire_smoke/train")
    ap.add_argument("--dup-bits", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.10)
    ap.add_argument("--clean-share", type=float, default=0.3, help="fraction of non-firing images also added")
    ap.add_argument("--report", default="benchmarks/results/hard_negatives.json")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="file names to drop after visual review (e.g. a lit fireplace is real fire, not a negative)")
    a = ap.parse_args()
    bad_words = ("fire", "smoke", "flame", "burn", "candle", "blaze")
    imgs = sorted(p for s in a.src for p in glob.glob(os.path.join(s, "**", "*"), recursive=True)
                  if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
                  and os.path.basename(p) not in set(a.exclude)
                  and not any(w in os.path.basename(p).lower() for w in bad_words))
    m = json.load(open(a.manifest))["eval"]
    ev = [p for v in m.values() for p, _ in v]
    E = np.stack([h for h in (phash(p) for p in ev) if h is not None])
    H, keep = [], []
    for p in imgs:
        h = phash(p)
        if h is not None:
            H.append(h)
            keep.append(p)
    d = min_dist(np.stack(H), E)
    cand = [p for p, dd in zip(keep, d) if dd > a.dup_bits]
    det = OnnxYolo(a.model, ["smoke", "fire"], conf=0.01)
    hard, clean = [], []
    for p in cand:
        img = cv2.imread(p)
        s = max([float(x.score) for x in det(img).detections], default=0.0)
        (hard if s >= a.min_score else clean).append((p, s))
    random.Random(0).shuffle(clean)
    chosen = hard + clean[:int(len(clean) * a.clean_share)]
    di, dl = os.path.join(a.out, "images"), os.path.join(a.out, "labels")
    for i, (p, _) in enumerate(chosen):
        name = f"hn_{i:04d}_{os.path.basename(p).replace(' ', '_')}"
        dst = os.path.join(di, name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(p), dst)
        open(os.path.join(dl, name.rsplit(".", 1)[0] + ".txt"), "w").close()
    rep = {"source_images": len(imgs), "removed_near_duplicates_of_eval": len(keep) - len(cand),
           "detector_fired": len(hard), "added_hard": len(hard), "added_clean": len(chosen) - len(hard),
           "min_score": a.min_score, "hard_examples": sorted([(os.path.basename(p), round(s, 3)) for p, s in hard], key=lambda x: -x[1])[:40]}
    json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "hard_examples"}, indent=1))


if __name__ == "__main__":
    main()
