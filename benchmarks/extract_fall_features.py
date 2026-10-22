"""Run pose estimation over GMDCSA24 once and cache raw detections + key-points.

Pose inference is the expensive part, so it is done once; the fusion model,
thresholds and calibration are then evaluated (and fitted cross-subject) on
the cached cues in seconds.

    python benchmarks/extract_fall_features.py --root /path/to/GMDCSA24 --out data/gmdcsa24_cues.pkl --stride 2
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgecopilot.io.video import VideoSource  # noqa: E402
from edgecopilot.perception.detector import OnnxYolo  # noqa: E402


def parse_intervals(classes: str):
    """'Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]' -> [('Falling (SW)', 3.4, 6.0), ...]"""
    out = []
    for name, a, b in re.findall(r"([^;\[\]]+?)\s*\[\s*([\d.]+)\s*to\s*([\d.]+)\s*\]", classes):
        out.append((name.strip(), float(a), float(b)))
    return out


def load_labels(root: str):
    items = []
    for subj in sorted(glob.glob(os.path.join(root, "Subject *"))):
        sid = int(subj.split()[-1])
        for kind in ("Fall", "ADL"):
            meta = {}
            with open(os.path.join(subj, f"{kind}.csv"), newline="", encoding="utf-8", errors="ignore") as f:
                for row in csv.reader(f):
                    if not row or row[0].strip().lower().startswith("file"):
                        continue
                    meta[row[0].strip()] = (row[4].strip() if len(row) > 4 else "", row[-1].strip())
            for vid in sorted(glob.glob(os.path.join(subj, kind, "*.mp4"))):
                name = os.path.basename(vid)
                desc, classes = meta.get(name, ("", ""))
                iv = parse_intervals(classes)
                fall_iv = [x for x in iv if x[0].lower().startswith("fall")]
                items.append({"path": vid, "subject": sid, "label": int(kind == "Fall"), "desc": desc,
                              "intervals": iv, "fall_onset": min((x[1] for x in fall_iv), default=None)})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/claude/datasets/gmdcsa24")
    ap.add_argument("--model", default="models/yolo11n-pose.onnx")
    ap.add_argument("--out", default="data/gmdcsa24_cues.pkl")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    items = load_labels(args.root)
    print(f"{len(items)} videos ({sum(i['label'] for i in items)} falls)")
    pose = OnnxYolo(args.model, ["person"], imgsz=args.imgsz, conf=0.35, iou=0.5, pose=True, threads=2)
    t0 = time.time()
    for k, it in enumerate(items):
        frames = []
        for frame, t in VideoSource(it["path"], stride=args.stride):
            out = pose(frame)
            # cache raw detections: tracking + cue extraction are re-run offline (cheap)
            frames.append({"t": t, "ms": out.latency_ms,
                           "dets": [(d.box.astype("float32"), float(d.score), d.keypoints.astype("float32"))
                                    for d in out.detections]})
        it["frames"] = frames
        it["shape"] = frame.shape[:2]
        print(f"[{k + 1}/{len(items)}] S{it['subject']} {'FALL' if it['label'] else 'ADL '} {os.path.basename(it['path'])} "
              f"{len(frames)} frames, elapsed {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(items, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
