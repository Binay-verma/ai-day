"""Privacy-filter verification: can an independent face detector still find faces after anonymisation?

Ground truth = faces that YuNet (OpenCV Zoo face detector, an independent model - the copilot
itself never runs a face detector) finds on the raw frame. Then the same frame is passed
through the copilot (pose -> privacy filter) and YuNet runs again on what would leave the box.

Reported per mode (blur, skeleton): share of raw faces that are still detectable afterwards
(lower is better; the goal is 0), and how many raw faces the pose model did not even see.

    python benchmarks/eval_privacy.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgecopilot.pipeline import Copilot, load_config  # noqa: E402
from edgecopilot.privacy import anonymize  # noqa: E402

DS = os.environ.get("EDGECOPILOT_DATASETS", "/home/claude/datasets")
VIDEOS = [f"{DS}/test_videos/face-demographics-walking.mp4", f"{DS}/test_videos/classroom.mp4",
          f"{DS}/test_videos/people-detection.mp4", f"{DS}/gmdcsa24/Subject 1/ADL/05.mp4",
          f"{DS}/gmdcsa24/Subject 2/ADL/05.mp4", f"{DS}/gmdcsa24/Subject 3/ADL/05.mp4", f"{DS}/gmdcsa24/Subject 4/ADL/05.mp4"]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def faces(det, img, thr):
    det.setInputSize((img.shape[1], img.shape[0]))
    _, f = det.detect(img)
    return [] if f is None else [r[:4] for r in f if r[14] >= thr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yunet", default="models/aux/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--every", type=float, default=1.0, help="sample one frame every N seconds")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--out", default="benchmarks/results/privacy.json")
    a = ap.parse_args()
    det = cv2.FaceDetectorYN.create(a.yunet, "", (320, 320), a.thr, 0.3, 5000)
    cfg = load_config("configs/campus.yaml")
    res = {"faces_raw": 0, "frames": 0, "per_video": {}}
    modes = {"blur": {"still_detected": 0, "pose_missed": 0}, "skeleton": {"still_detected": 0, "pose_missed": 0},
             "no_filter_control": {"still_detected": 0}}
    for v in VIDEOS:
        if not os.path.exists(v):
            continue
        cop = Copilot(cfg, root=".", zone="privacy test", enable={"fall": True, "crowd": False, "fire_smoke": False})
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps * a.every)))
        i, nv = 0, 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if f.shape[1] > 960:
                f = cv2.resize(f, (960, int(f.shape[0] * 960 / f.shape[1])))
            r = cop.process(f, i / fps)
            i += 1
            if (i - 1) % step:
                continue
            gt = faces(det, f, a.thr)
            if not gt:
                continue
            res["frames"] += 1
            res["faces_raw"] += len(gt)
            nv += len(gt)
            people = r.privacy_people if hasattr(r, "privacy_people") else r.people
            for mode in ("blur", "skeleton"):
                out = anonymize(f, people, mode, pose_available=True)
                after = faces(det, out, a.thr)
                modes[mode]["still_detected"] += sum(1 for g in gt if any(iou(g, x) > 0.3 for x in after))
            boxes = [p.box for p in people]
            modes["blur"]["pose_missed"] += sum(1 for g in gt if not any(
                b[0] <= g[0] + g[2] / 2 <= b[2] and b[1] <= g[1] + g[3] / 2 <= b[3] for b in boxes))
            modes["no_filter_control"]["still_detected"] += len(gt)
        res["per_video"][os.path.basename(os.path.dirname(os.path.dirname(v))) + "/" + os.path.basename(v)] = nv
    n = max(res["faces_raw"], 1)
    for m in modes.values():
        m["rate"] = round(m["still_detected"] / n, 4)
    res["modes"] = modes
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
