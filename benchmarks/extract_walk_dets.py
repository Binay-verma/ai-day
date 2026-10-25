"""Cache person detections (YOLO11n-pose, shipped settings) for the walking videos used by eval_exits.py."""
from __future__ import annotations

import os
import pickle
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgecopilot.perception.detector import OnnxYolo  # noqa: E402

DS = os.environ.get("EDGECOPILOT_DATASETS", "/home/claude/datasets")
CLIPS = ["vtest.avi", "one-by-one-person-detection.mp4", "people-detection.mp4", "face-demographics-walking.mp4",
         "store-aisle-detection.mp4"]
TARGET_FPS = 10.0


def main(out="data/walk_people.pkl"):
    pose = OnnxYolo("models/yolo11n-pose.onnx", ["person"], conf=0.25, pose=True, threads=2)
    res = {}
    for name in CLIPS:
        cap = cv2.VideoCapture(f"{DS}/test_videos/{name}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        step = max(1, round(fps / TARGET_FPS))
        frames, i, shape = [], 0, None
        while True:
            ok, f = cap.read()
            if not ok:
                break
            shape = f.shape[:2]
            if i % step == 0:
                d = pose(f)
                frames.append((i / fps, [(det.box.copy(), float(det.score)) for det in d.detections]))
            i += 1
        res[name] = {"shape": shape, "fps": fps / step, "frames": frames}
        print(name, len(frames), flush=True)
    pickle.dump(res, open(out, "wb"))


if __name__ == "__main__":
    main()
