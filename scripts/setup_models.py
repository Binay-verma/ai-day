"""Download / export / quantise every model the edge runtime needs.

    pip install -e ".[train]"      # once, on any machine with internet
    python scripts/setup_models.py
    -> models/yolo11n-pose.onnx (+ .int8.onnx), models/fire_smoke.onnx (+ .int8.onnx)

The edge box itself only needs the .onnx files and onnxruntime.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt"


def fetch(url: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    print("downloading", url)
    urllib.request.urlretrieve(url, dst)


def run(*args: str) -> None:
    print("+", " ".join(args))
    subprocess.check_call([sys.executable, *args], cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire-weights", default=os.path.join(ROOT, "models", "fire_smoke.pt"),
                    help="fine-tuned fire/smoke checkpoint (see scripts/train_fire.py)")
    ap.add_argument("--no-int8", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    pose_pt = os.path.join(ROOT, "models", "yolo11n-pose.pt")
    fetch(POSE_URL, pose_pt)
    run("scripts/export_onnx.py", "--weights", pose_pt, "--out", "models/yolo11n-pose.onnx", "--feature", "p3")
    if os.path.exists(args.fire_weights):
        run("scripts/export_onnx.py", "--weights", args.fire_weights, "--out", "models/fire_smoke.onnx", "--feature", "p3")
    else:
        print("no fire/smoke checkpoint found - train one with scripts/train_fire.py (the copilot falls back to image + sensor cues)")
    if not args.no_int8:
        run("scripts/quantize.py", "--model", "models/yolo11n-pose.onnx")
        if os.path.exists(os.path.join(ROOT, "models", "fire_smoke.onnx")):
            run("scripts/quantize.py", "--model", "models/fire_smoke.onnx", "--calib", "data/fire_smoke/val/images")


if __name__ == "__main__":
    main()
