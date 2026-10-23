"""Fire/smoke detector benchmark: mAP@0.5 per class + image-level recall/FPR.

Evaluation sets (all held out from training):
* ``homefire_test``  Home-fire test split (indoor, 1,300 images, YOLO labels)
* ``dba_val``        our deterministic 20 % hold-out of DBA-Fire (hash split)
* ``neutral``        DeepQuest "Neutral" images (no fire/smoke) -> false-positive rate

Checkpoints with other class orders are remapped by class *name*.  Works with
Ultralytics .pt checkpoints (screening/training) and with our ONNX runtime
(``--onnx``) so the deployed FP32 and INT8 models are scored on the same data.

    python benchmarks/eval_fire.py --weights a.pt b.pt --limit 400
    python benchmarks/eval_fire.py --onnx models/fire_smoke.onnx models/fire_smoke.int8.onnx
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DS = "/home/claude/datasets"
CLASSES = ["fire", "smoke"]


def canon(name: str):
    n = name.lower()
    if "smoke" in n:
        return "smoke"
    if "fire" in n or "flame" in n:
        return "fire"
    return None


def dba_split(path: str) -> str:
    h = int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16) % 10
    return "val" if h < 2 else "train"


def read_labels(lbl_path: str, w: int, h: int, names=("fire", "smoke")):
    out = []
    if os.path.exists(lbl_path):
        for line in open(lbl_path):
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            out.append((names[c], np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])))
    return out


def eval_sets(limit: int, seed: int = 0):
    rng = random.Random(seed)
    sets = {}
    hf = sorted(glob.glob(f"{DS}/home_fire/test/images/*.jpg"))
    rng.shuffle(hf)
    sets["homefire_test"] = [(p, p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt") for p in hf[:limit]]
    dba = [p for p in sorted(glob.glob(f"{DS}/dba_fire/repo/JPEGImages/*")) if dba_split(p) == "val"]
    rng.shuffle(dba)
    sets["dba_val"] = [(p, p.replace("/JPEGImages/", "/txt/").rsplit(".", 1)[0] + ".txt") for p in dba[:limit]]
    neu = sorted(glob.glob(f"{DS}/deepquest_fire_smoke/x/FIRE-SMOKE-DATASET/Test/Neutral/*"))
    # the first 600 sorted Train/Neutral images are training negatives; evaluate on the last 300 only
    neu += sorted(glob.glob(f"{DS}/deepquest_fire_smoke/x/FIRE-SMOKE-DATASET/Train/Neutral/*"))[-300:][:max(0, limit - 100)]
    sets["neutral"] = [(p, None) for p in neu[:limit]]
    return sets


def ap50(dets, gts, cls):
    """VOC all-point AP at IoU 0.5 for one class. dets: [(img, score, box)], gts: {img: [box]}."""
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return float("nan")
    dets = sorted(dets, key=lambda d: -d[1])
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = np.zeros(len(dets))
    for i, (img, s, b) in enumerate(dets):
        g = gts.get(img, [])
        if not len(g):
            continue
        G = np.array(g)
        ix1, iy1 = np.maximum(G[:, 0], b[0]), np.maximum(G[:, 1], b[1])
        ix2, iy2 = np.minimum(G[:, 2], b[2]), np.minimum(G[:, 3], b[3])
        inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
        iou = inter / ((G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
        j = int(iou.argmax())
        if iou[j] >= 0.5 and not used[img][j]:
            tp[i] = 1
            used[img][j] = True
    fp = 1 - tp
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec = tpc / npos
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class PtPredictor:
    def __init__(self, path, imgsz=640):
        from ultralytics import YOLO
        self.m = YOLO(path)
        self.map = {i: canon(n) for i, n in self.m.names.items()}
        self.imgsz = imgsz

    def __call__(self, img):
        r = self.m.predict(img, imgsz=self.imgsz, conf=0.001, iou=0.6, verbose=False, max_det=100)[0]
        out = []
        for b, s, c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy().astype(int)):
            k = self.map.get(c)
            if k:
                out.append((k, float(s), b))
        return out


class OnnxPredictor:
    def __init__(self, path, names=("fire", "smoke")):
        from edgecopilot.perception.detector import OnnxYolo
        self.m = OnnxYolo(path, list(names), conf=0.001, iou=0.6, threads=2)

    def __call__(self, img):
        return [(canon(d.label), d.score, d.box) for d in self.m(img).detections if canon(d.label)]


def evaluate(pred, sets, conf_thr=0.25):
    res = {}
    for name, items in sets.items():
        dets = {c: [] for c in CLASSES}
        gts = {c: {} for c in CLASSES}
        img_pos, img_pred, ms = [], [], []
        for p, lbl in items:
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            g = read_labels(lbl, w, h) if lbl else []
            for c in CLASSES:
                gts[c][p] = [b for k, b in g if k == c]
            t0 = time.perf_counter()
            out = pred(img)
            ms.append((time.perf_counter() - t0) * 1e3)
            for k, s, b in out:
                dets[k].append((p, s, b))
            img_pos.append(bool(g))
            img_pred.append(any(s >= conf_thr for k, s, b in out))
        aps = {c: ap50(dets[c], gts[c], c) for c in CLASSES}
        valid = [v for v in aps.values() if v == v]
        img_pos, img_pred = np.array(img_pos), np.array(img_pred)
        r = {"n": len(img_pos), "ms_per_image": float(np.median(ms)) if ms else None}
        if valid:
            r.update({"ap50_fire": aps["fire"], "ap50_smoke": aps["smoke"], "map50": float(np.mean(valid))})
        if img_pos.any():
            r["image_recall@%.2f" % conf_thr] = float(img_pred[img_pos].mean())
        if (~img_pos).any():
            r["image_fpr@%.2f" % conf_thr] = float(img_pred[~img_pos].mean())
        res[name] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="*", default=[])
    ap.add_argument("--onnx", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--onnx-names", default="smoke,fire", help="class order of the ONNX model(s)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="benchmarks/results/fire_eval.json")
    args = ap.parse_args()
    sets = eval_sets(args.limit)
    print({k: len(v) for k, v in sets.items()})
    allres = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for w in args.weights:
        key = w.split("fire_smoke/")[-1]
        try:
            allres[key] = evaluate(PtPredictor(w, args.imgsz), sets)
        except Exception as e:  # noqa: BLE001
            allres[key] = {"error": str(e)}
        print(key, json.dumps(allres[key]), flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(allres, open(args.out, "w"), indent=2)
    for w in args.onnx:
        allres[w] = evaluate(OnnxPredictor(w, tuple(args.onnx_names.split(","))), sets)
        print(w, json.dumps(allres[w]), flush=True)
        json.dump(allres, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
