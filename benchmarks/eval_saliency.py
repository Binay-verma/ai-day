"""Faithfulness of the visual evidence maps for the fire/smoke detector.

Deletion / insertion metrics (Petsiuk et al., RISE 2018) adapted to a detector
(D-RISE style score): for a target detection we progressively remove (deletion,
grey fill) or reveal (insertion, starting from a blurred image) pixels in order
of saliency and track the score of the best same-class detection overlapping
the original box (IoU >= 0.5).  Lower deletion AUC / higher insertion AUC =
more faithful.  Compared maps:

* ``eigencam``      gradient-free Eigen-CAM from the exported P3 neck feature (what runs on the box)
* ``eigencam_box``  the same, re-normalised inside the detection box (what the console shows)
* ``occlusion``     model-agnostic 6x6 occlusion map (slow reference, 36 extra inferences)
* ``box_uniform``   uniform inside the box  -> control: does the CAM add anything beyond the box?
* ``center``        centred Gaussian        -> control
* ``random``        uniform noise           -> control
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgecopilot.perception.detector import OnnxYolo  # noqa: E402
from edgecopilot.perception.tracker import iou_matrix  # noqa: E402
from edgecopilot.xai.saliency import box_renormalize, eigencam, occlusion_saliency  # noqa: E402


def target_score(model, img, box, cls):
    dets = [d for d in model(img).detections if d.cls == cls]
    if not dets:
        return 0.0
    iou = iou_matrix(np.array([box]), np.array([d.box for d in dets]))[0]
    best = [d.score * (iou[i] >= 0.5) for i, d in enumerate(dets)]
    return float(max(best))


def curve(model, img, sal, box, cls, mode, steps=20):
    h, w = img.shape[:2]
    order = np.argsort(-(sal + 1e-6 * np.random.default_rng(0).random(sal.shape)).ravel())
    blurred = cv2.GaussianBlur(img, (51, 51), 25)
    grey = np.full_like(img, 114)
    scores = []
    for k in range(steps + 1):
        n = int(len(order) * k / steps)
        mask = np.zeros(h * w, bool)
        mask[order[:n]] = True
        mask = mask.reshape(h, w)[..., None]
        if mode == "deletion":
            x = np.where(mask, grey, img)
        else:
            x = np.where(mask, img, blurred)
        scores.append(target_score(model, x.astype(np.uint8), box, cls))
    s = np.array(scores)
    return float(np.trapezoid(s, dx=1 / steps)), s.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fire_smoke.onnx")
    ap.add_argument("--names", default="smoke,fire")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="benchmarks/results/saliency.json")
    args = ap.parse_args()
    model = OnnxYolo(args.model, args.names.split(","), conf=0.05, iou=0.5, threads=2)
    imgs = sorted(glob.glob("/home/claude/datasets/home_fire/test/images/*.jpg"))
    random.Random(1).shuffle(imgs)
    methods = ["eigencam", "eigencam_box", "occlusion", "box_uniform", "center", "random"]
    res = {m: {"deletion": [], "insertion": [], "ms": []} for m in methods}
    used = 0
    for p in imgs:
        if used >= args.n:
            break
        img = cv2.imread(p)
        out = model(img)
        dets = [d for d in out.detections if d.score >= 0.4]
        if not dets or out.feature is None:
            continue
        d = max(dets, key=lambda x: x.score)
        h, w = img.shape[:2]
        maps = {}
        t0 = time.perf_counter()
        maps["eigencam"] = eigencam(out.feature, out.letterbox, img.shape)
        res["eigencam"]["ms"].append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        maps["eigencam_box"] = box_renormalize(maps["eigencam"], [d.box])
        res["eigencam_box"]["ms"].append((time.perf_counter() - t0) * 1e3 + res["eigencam"]["ms"][-1])
        t0 = time.perf_counter()
        maps["occlusion"] = occlusion_saliency(img, lambda x: target_score(model, x, d.box, d.cls), grid=6)
        res["occlusion"]["ms"].append((time.perf_counter() - t0) * 1e3)
        bu = np.zeros((h, w), np.float32)
        x1, y1, x2, y2 = d.box.astype(int)
        bu[max(0, y1):y2, max(0, x1):x2] = 1.0
        maps["box_uniform"] = bu
        yy, xx = np.mgrid[0:h, 0:w]
        maps["center"] = np.exp(-(((xx - w / 2) / (w / 4)) ** 2 + ((yy - h / 2) / (h / 4)) ** 2))
        maps["random"] = np.random.default_rng(used).random((h, w)).astype(np.float32)
        for m in methods:
            if m not in ("eigencam", "eigencam_box", "occlusion"):
                res[m]["ms"].append(0.0)
            dauc, _ = curve(model, img, maps[m], d.box, d.cls, "deletion", args.steps)
            iauc, _ = curve(model, img, maps[m], d.box, d.cls, "insertion", args.steps)
            res[m]["deletion"].append(dauc)
            res[m]["insertion"].append(iauc)
        used += 1
        print(f"[{used}/{args.n}] " + " ".join(f"{m}: del {np.mean(res[m]['deletion']):.3f} ins {np.mean(res[m]['insertion']):.3f}" for m in methods), flush=True)
    summary = {m: {"deletion_auc_mean": float(np.mean(v["deletion"])), "insertion_auc_mean": float(np.mean(v["insertion"])),
                   "ms_median": float(np.median(v["ms"])), "n": len(v["deletion"])} for m, v in res.items()}
    # paired comparison eigencam_box vs box_uniform (does the CAM add information inside the box?)
    a, b = np.array(res["eigencam_box"]["deletion"]), np.array(res["box_uniform"]["deletion"])
    summary["eigencam_box_beats_box_uniform_deletion_frac"] = float(np.mean(a < b))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "per_image": res}, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
