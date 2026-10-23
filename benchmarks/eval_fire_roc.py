"""Image-level operating curve of the fire/smoke detector (leakage-free v2 sets).

An image's score is its highest fire/smoke detection confidence. Positives: Home-fire
test images with at least one box; negatives: DeepQuest Neutral (never trained on).
Reports recall at the plan's 10 % false-positive budget, the false-positive rate at
85 % recall, and the threshold for each, so a site can choose its operating point.

    python benchmarks/eval_fire_roc.py --onnx models/fire_smoke.onnx
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgecopilot.perception.detector import OnnxYolo  # noqa: E402


def scores(det, items):
    out = []
    for img_path, lbl in items:
        img = cv2.imread(img_path)
        if img is None:
            continue
        o = det(img)
        s = max([float(d.score) for d in o.detections], default=0.0)
        pos = bool(lbl) and os.path.exists(lbl) and os.path.getsize(lbl) > 0
        out.append((s, pos))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/fire_smoke.onnx")
    ap.add_argument("--manifest", default="data/fire_splits.json")
    ap.add_argument("--out", default="benchmarks/results/fire_roc.json")
    a = ap.parse_args()
    m = json.load(open(a.manifest))["eval"]
    det = OnnxYolo(a.onnx, ["smoke", "fire"], conf=0.01)
    hf = scores(det, [tuple(x) for x in m["homefire_test"]])
    neu = scores(det, [(p, None) for p, _ in m["neutral"]])
    pos = np.array([s for s, p in hf if p])
    neg = np.array([s for s, _ in neu])
    ths = np.unique(np.concatenate([pos, neg, [1.01]]))
    tpr = np.array([(pos >= t).mean() for t in ths])
    fpr = np.array([(neg >= t).mean() for t in ths])
    ok = fpr <= 0.10
    i10 = np.argmax(np.where(ok, tpr, -1))
    ok85 = tpr >= 0.85
    j85 = np.argmin(np.where(ok85, fpr, 2))
    order = np.argsort(fpr)
    auc = float(np.trapezoid(tpr[order], fpr[order])) if hasattr(np, "trapezoid") else float(np.trapz(tpr[order], fpr[order]))
    res = {"model": a.onnx, "n_pos": int(len(pos)), "n_neg": int(len(neg)), "roc_auc": auc,
           "recall_at_fpr10": float(tpr[i10]), "threshold_at_fpr10": float(ths[i10]), "fpr_at_that_threshold": float(fpr[i10]),
           "fpr_at_recall85": float(fpr[j85]) if ok85.any() else None, "threshold_at_recall85": float(ths[j85]) if ok85.any() else None,
           "at_0.25": {"recall": float((pos >= 0.25).mean()), "fpr": float((neg >= 0.25).mean())}}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
