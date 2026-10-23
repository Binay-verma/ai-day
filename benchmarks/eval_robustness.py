"""Corruption robustness of the two detectors (ImageNet-C / Michaelis et al. 2019 protocol).

CCTV-relevant subset of the ImageNet-C corruptions (Hendrycks & Dietterich 2019) plus low
light, implemented with OpenCV/NumPy (no extra dependency), severities 1, 3, 5 with
parameters following the `imagecorruptions` package where one exists:

    gaussian_noise, motion_blur, defocus_blur, fog, contrast, jpeg_compression, pixelate, low_light

* Fire/smoke detector: mAP@0.5 on a fixed random sample of the leakage-free Home-fire test set
  and false-positive rate on the DeepQuest neutral set; reported as mPC (mean over
  corruptions x severities) and rPC = mPC / clean (Michaelis et al.).
* Person/pose detector: recall of the clean-image person detections (score >= 0.5, the
  pseudo ground truth) under each corruption, on frames sampled from GMDCSA24 ADL clips and
  the Intel sample videos - i.e. "does the system still see the people it saw before?".

    python benchmarks/eval_robustness.py --n-fire 300
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
from eval_fire import ap50, read_labels  # noqa: E402

from edgecopilot.perception.detector import OnnxYolo  # noqa: E402

DS = os.environ.get("EDGECOPILOT_DATASETS", "/home/claude/datasets")


def _rng(img):
    return np.random.default_rng(int(img[::17, ::17].sum()) % (2**32))


def gaussian_noise(img, s):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][s - 1]
    x = img.astype(np.float32) / 255
    return (np.clip(x + _rng(img).normal(0, c, x.shape), 0, 1) * 255).astype(np.uint8)


def motion_blur(img, s):
    k = [7, 11, 15, 21, 27][s - 1]
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), 30, 1.0)
    ker = cv2.warpAffine(ker, m, (k, k))
    return cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))


def defocus_blur(img, s):
    r = [3, 4, 6, 8, 10][s - 1]
    ker = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(ker, (r, r), r, 1.0, -1)
    return cv2.filter2D(img, -1, ker / ker.sum())


def fog(img, s):
    a = [0.25, 0.35, 0.45, 0.55, 0.65][s - 1]
    h, w = img.shape[:2]
    noise = cv2.resize(_rng(img).random((max(2, h // 32), max(2, w // 32))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    haze = (0.75 + 0.25 * noise)[..., None] * 255
    return np.clip(img * (1 - a) + haze * a, 0, 255).astype(np.uint8)


def contrast(img, s):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][s - 1]
    m = img.mean(axis=(0, 1), keepdims=True)
    return np.clip((img - m) * c + m, 0, 255).astype(np.uint8)


def jpeg_compression(img, s):
    q = [25, 18, 15, 10, 7][s - 1]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def pixelate(img, s):
    f = [0.6, 0.5, 0.4, 0.3, 0.25][s - 1]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def low_light(img, s):
    """Night without IR: darker (gamma + gain) and noisier (shot noise grows as light falls)."""
    gain, gamma, noise = [(0.6, 1.4, 0.02), (0.45, 1.7, 0.03), (0.3, 2.0, 0.04), (0.2, 2.3, 0.05), (0.12, 2.6, 0.06)][s - 1]
    x = (img.astype(np.float32) / 255) ** gamma * gain
    x = x + _rng(img).normal(0, noise, x.shape)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


CORRUPTIONS = {f.__name__: f for f in (gaussian_noise, motion_blur, defocus_blur, fog, contrast, jpeg_compression, pixelate, low_light)}
SEVERITIES = (1, 3, 5)


def fire_eval(det, pos, neg, fn=None, s=0):
    gts = {"fire": {}, "smoke": {}}
    preds = {"fire": [], "smoke": []}
    for k, (p, lbl) in enumerate(pos):
        img = cv2.imread(p)
        h, w = img.shape[:2]
        x = fn(img, s) if fn else img
        for name, box in read_labels(lbl, w, h):
            gts[name].setdefault(k, []).append(box)
        for d in det(x).detections:
            if d.label in preds:
                preds[d.label].append((k, float(d.score), d.box))
    vals = [ap50(preds[c], gts[c], c) for c in ("fire", "smoke")]
    m = float(np.nanmean(vals))
    fp = 0
    for p, _ in neg:
        img = cv2.imread(p)
        x = fn(img, s) if fn else img
        fp += int(any(d.score >= 0.25 for d in det(x).detections))
    return m, fp / max(len(neg), 1)


def person_eval(det, frames, fn=None, s=0, ref=None):
    found, total, out = 0, 0, []
    for i, img in enumerate(frames):
        x = fn(img, s) if fn else img
        boxes = [d.box for d in det(x).detections if d.score >= (0.5 if ref is None else 0.25)]
        out.append(boxes)
        if ref is not None:
            for g in ref[i]:
                total += 1
                found += int(any(_iou(g, b) >= 0.5 for b in boxes))
    return out, (found / total if total else None)


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def sample_frames(n_per_video=12):
    vids = sorted(glob.glob(f"{DS}/gmdcsa24/Subject */ADL/0[1-4].mp4")) + [f"{DS}/test_videos/{v}" for v in
                                                                             ("classroom.mp4", "store-aisle-detection.mp4", "people-detection.mp4")]
    frames = []
    for v in vids:
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for idx in np.linspace(0, max(n - 1, 0), n_per_video).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                if f.shape[1] > 960:
                    f = cv2.resize(f, (960, int(f.shape[0] * 960 / f.shape[1])))
                frames.append(f)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", default="models/fire_smoke.onnx")
    ap.add_argument("--pose", default="models/yolo11n-pose.onnx")
    ap.add_argument("--n-fire", type=int, default=300)
    ap.add_argument("--n-neg", type=int, default=200)
    ap.add_argument("--out", default="benchmarks/results/robustness.json")
    a = ap.parse_args()
    m = json.load(open("data/fire_splits.json"))["eval"]
    rng = random.Random(0)
    pos = rng.sample([tuple(x) for x in m["homefire_test"]], a.n_fire)
    neg = rng.sample([(p, None) for p, _ in m["neutral"]], a.n_neg)
    fire = OnnxYolo(a.fire, ["smoke", "fire"], conf=0.01)
    pose = OnnxYolo(a.pose, ["person"], pose=True, conf=0.25)
    frames = sample_frames()
    res = {"severities": SEVERITIES, "n_fire_images": len(pos), "n_neutral": len(neg), "n_person_frames": len(frames),
           "fire": {}, "person": {}}
    clean_map, clean_fpr = fire_eval(fire, pos, neg)
    ref, _ = person_eval(pose, frames)
    res["fire"]["clean"] = {"map50": clean_map, "fpr": clean_fpr}
    res["person"]["clean"] = {"recall": 1.0, "n_people": sum(len(r) for r in ref)}
    print("clean", round(clean_map, 3), round(clean_fpr, 3), flush=True)
    for name, fn in CORRUPTIONS.items():
        for s in SEVERITIES:
            mm, fpr = fire_eval(fire, pos, neg, fn, s)
            _, rec = person_eval(pose, frames, fn, s, ref)
            res["fire"][f"{name}_{s}"] = {"map50": mm, "fpr": fpr}
            res["person"][f"{name}_{s}"] = {"recall": rec}
            print(f"{name:17s} s{s}: fire mAP {mm:.3f} FPR {fpr:.3f} | person recall {rec:.3f}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    keys = [k for k in res["fire"] if k != "clean"]
    mpc = float(np.mean([res["fire"][k]["map50"] for k in keys]))
    prec = float(np.mean([res["person"][k]["recall"] for k in keys]))
    res["summary"] = {"fire_mPC": mpc, "fire_rPC": mpc / clean_map, "person_mean_recall_under_corruption": prec,
                      "worst_fire": min(keys, key=lambda k: res["fire"][k]["map50"]),
                      "worst_person": min(keys, key=lambda k: res["person"][k]["recall"])}
    print(json.dumps(res["summary"], indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
