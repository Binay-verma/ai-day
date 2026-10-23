"""Leakage-free fire/smoke train / evaluation split (v2).

The first protocol split DBA-Fire by file-name hash and used the upstream
Home-fire train/test split as-is.  An independent audit found that both leak:
DBA images are frames sampled from videos, so neighbouring frames landed on
both sides, and the Home-fire test set itself contains (near-)duplicates of
training images (18.5 % exact pHash duplicates, 58 % within 8 bits).

This version:

1. computes a 64-bit DCT perceptual hash (pHash) for every image;
2. groups DBA-Fire images into *scenes* (union-find over pairs within
   ``--dup-bits`` Hamming distance), and splits whole scenes 80/20;
3. defines the evaluation sets (Home-fire test, DBA scene hold-out, DeepQuest
   Neutral test negatives) and then **removes from training and validation
   every image within ``--dup-bits`` of any evaluation image**;
4. writes the manifest ``data/fire_splits.json`` that ``benchmarks/eval_fire.py``
   uses, plus a YOLO dataset with symlinks under ``data/fire_smoke``.

Class order in the written labels: 0 = smoke, 1 = fire (order of the D-Fire
start checkpoint). Source datasets use 0 = fire, 1 = smoke and are swapped here.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil

import cv2
import numpy as np

DS = "/home/claude/datasets"
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def phash(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(img)[:8, :8].flatten()
    return np.packbits(d > np.median(d[1:]))


def hash_all(paths, cache):
    c = {}
    if os.path.exists(cache):
        c = dict(np.load(cache, allow_pickle=True)["h"].item())
    for p in paths:
        if p not in c:
            c[p] = phash(p)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, h=np.array(c, dtype=object))
    keep = [p for p in paths if c.get(p) is not None]
    return keep, (np.stack([c[p] for p in keep]) if keep else np.zeros((0, 8), np.uint8))


def min_dist(Q, R, chunk=256):
    out = np.full(len(Q), 64, int)
    if len(R) == 0:
        return out
    for i in range(0, len(Q), chunk):
        d = POP[np.bitwise_xor(Q[i:i + chunk, None, :], R[None, :, :])].sum(-1)
        out[i:i + chunk] = d.min(1)
    return out


def scenes(H, bits):
    """Union-find connected components of images within `bits` Hamming distance."""
    parent = list(range(len(H)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(0, len(H), 256):
        d = POP[np.bitwise_xor(H[i:i + 256, None, :], H[None, :, :])].sum(-1)
        for a, b in zip(*np.nonzero(d <= bits)):
            a += i
            if a < b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    return [find(i) for i in range(len(H))]


def label_path(img: str) -> str:
    if "/JPEGImages/" in img:
        return img.replace("/JPEGImages/", "/txt/").rsplit(".", 1)[0] + ".txt"
    return img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"


def write_yolo(out_root, split, items):
    di, dl = os.path.join(out_root, split, "images"), os.path.join(out_root, split, "labels")
    os.makedirs(di, exist_ok=True)
    os.makedirs(dl, exist_ok=True)
    for prefix, img, lbl in items:
        name = f"{prefix}_{os.path.basename(img)}"
        os.symlink(os.path.abspath(img), os.path.join(di, name))
        lines = []
        if lbl and os.path.exists(lbl):
            for line in open(lbl):
                p = line.split()
                if len(p) >= 5:
                    p[0] = "1" if p[0] == "0" else "0"  # source 0=fire,1=smoke -> 0=smoke,1=fire
                    lines.append(" ".join(p))
        with open(os.path.join(dl, name.rsplit(".", 1)[0] + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fire_smoke")
    ap.add_argument("--manifest", default="data/fire_splits.json")
    ap.add_argument("--dup-bits", type=int, default=10)
    ap.add_argument("--n-neg", type=int, default=600)
    args = ap.parse_args()
    cache = "data/phash_cache.npz"
    hf_tr = sorted(glob.glob(f"{DS}/home_fire/train/images/*"))
    hf_va = sorted(glob.glob(f"{DS}/home_fire/val/images/*"))
    hf_te = sorted(glob.glob(f"{DS}/home_fire/test/images/*"))
    dba = sorted(glob.glob(f"{DS}/dba_fire/repo/JPEGImages/*"))
    neu_tr = sorted(glob.glob(f"{DS}/deepquest_fire_smoke/x/FIRE-SMOKE-DATASET/Train/Neutral/*"))
    neu_te = sorted(glob.glob(f"{DS}/deepquest_fire_smoke/x/FIRE-SMOKE-DATASET/Test/Neutral/*"))
    allp = hf_tr + hf_va + hf_te + dba + neu_tr + neu_te
    keep, H = hash_all(allp, cache)
    idx = {p: i for i, p in enumerate(keep)}
    h = lambda ps: H[[idx[p] for p in ps if p in idx]] if ps else np.zeros((0, 8), np.uint8)  # noqa: E731

    # DBA: split by scene, not by file
    dba = [p for p in dba if p in idx]
    roots = scenes(h(dba), args.dup_bits)
    scene_ids = sorted(set(roots))
    val_scenes = {r for r in scene_ids if int(hashlib.md5(str(min(i for i, x in enumerate(roots) if x == r)).encode()).hexdigest(), 16) % 5 == 0}
    dba_val = [p for p, r in zip(dba, roots) if r in val_scenes]
    dba_tr = [p for p, r in zip(dba, roots) if r not in val_scenes]

    # evaluation sets
    neg_eval = neu_te + neu_tr[args.n_neg:]  # never-trained negatives
    evals = {"homefire_test": hf_te, "dba_val": dba_val, "neutral": neg_eval}
    E = np.concatenate([h(v) for v in evals.values()])
    # training/validation candidates, minus anything near any evaluation image
    cand_tr = [("hf", p) for p in hf_tr] + [("dba", p) for p in dba_tr] + [("neg", p) for p in neu_tr[:args.n_neg]]
    cand_va = [("hf", p) for p in hf_va]
    d_tr = min_dist(h([p for _, p in cand_tr]), E)
    d_va = min_dist(h([p for _, p in cand_va]), E)
    train = [(pre, p) for (pre, p), d in zip(cand_tr, d_tr) if d > args.dup_bits]
    val = [(pre, p) for (pre, p), d in zip(cand_va, d_va) if d > args.dup_bits]
    removed_tr = len(cand_tr) - len(train)
    removed_va = len(cand_va) - len(val)
    # remaining overlap check (must be zero by construction)
    resid = int((min_dist(h([p for _, p in train]), E) <= args.dup_bits).sum())
    assert resid == 0

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    write_yolo(args.out, "train", [(pre, p, None if pre == "neg" else label_path(p)) for pre, p in train])
    write_yolo(args.out, "val", [(pre, p, label_path(p)) for pre, p in val])
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: train/images\nval: val/images\nnames:\n  0: smoke\n  1: fire\n")
    stats = {"dup_bits": args.dup_bits, "dba_scenes": len(scene_ids), "dba_val_scenes": len(val_scenes),
             "train": len(train), "val": len(val), "removed_from_train_as_near_duplicates_of_eval": removed_tr,
             "removed_from_val": removed_va, "eval_sizes": {k: len(v) for k, v in evals.items()}}
    json.dump({"protocol": "v2-leakage-free", "stats": stats,
               "eval": {k: [[p, None if k == "neutral" else label_path(p)] for p in v] for k, v in evals.items()}},
              open(args.manifest, "w"), indent=0)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
