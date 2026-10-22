"""Assemble the fire/smoke training set (YOLO format, symlinks, no copies).

train = Home-fire train (indoor, 3,900) + DBA-Fire 80 % hash split (outdoor/
        public spaces) + DeepQuest Neutral *train* images as hard negatives
        (empty label files)
val   = Home-fire val (1,300)
Test sets used by benchmarks/eval_fire.py (Home-fire test, DBA 20 %,
DeepQuest Neutral *test*) are never touched here.
Classes: 0 = fire, 1 = smoke (both source datasets already use this order).
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.eval_fire import dba_split  # noqa: E402

DS = "/home/claude/datasets"
OUT = os.environ.get("FIRE_DATA", "data/fire_smoke")


def link(src_img, src_lbl, split, prefix):
    name = f"{prefix}_{os.path.basename(src_img)}"
    di, dl = os.path.join(OUT, split, "images"), os.path.join(OUT, split, "labels")
    os.makedirs(di, exist_ok=True)
    os.makedirs(dl, exist_ok=True)
    dst = os.path.join(di, name)
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(src_img), dst)
    lbl_dst = os.path.join(dl, name.rsplit(".", 1)[0] + ".txt")
    with open(lbl_dst, "w") as f:
        if src_lbl and os.path.exists(src_lbl):
            f.write(open(src_lbl).read())


def main():
    n = {"train": 0, "val": 0}
    for p in glob.glob(f"{DS}/home_fire/train/images/*"):
        link(p, p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt", "train", "hf"); n["train"] += 1
    for p in glob.glob(f"{DS}/dba_fire/repo/JPEGImages/*"):
        if dba_split(p) == "train":
            link(p, p.replace("/JPEGImages/", "/txt/").rsplit(".", 1)[0] + ".txt", "train", "dba"); n["train"] += 1
    for p in sorted(glob.glob(f"{DS}/deepquest_fire_smoke/x/FIRE-SMOKE-DATASET/Train/Neutral/*"))[:600]:
        link(p, None, "train", "neg"); n["train"] += 1
    for p in glob.glob(f"{DS}/home_fire/val/images/*"):
        link(p, p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt", "val", "hf"); n["val"] += 1
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(OUT)}\ntrain: train/images\nval: val/images\nnames:\n  0: fire\n  1: smoke\n")
    print(n, "->", OUT)


if __name__ == "__main__":
    main()
