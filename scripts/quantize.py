"""Static INT8 (QDQ) quantisation of a YOLO ONNX model for CPU edge inference.

Follows ONNX Runtime guidance for CNNs: static quantisation, QDQ format,
per-channel weights, calibration on real frames, and the detection head (box
decoding / DFL / concat / sigmoid) kept in FP32 because its value ranges are
too different to share one INT8 scale (head exclusion recovers most of the
mAP lost by naive full-INT8 YOLO quantisation).

    python scripts/quantize.py --model models/fire_smoke.onnx --calib data/fire_smoke/val/images
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process


def letterbox(img, size):
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    r = cv2.resize(img, (int(round(w * s)), int(round(h * s))))
    out = np.full((size, size, 3), 114, np.uint8)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


class Reader(CalibrationDataReader):
    def __init__(self, paths, input_name, size):
        self.paths = paths
        self.input_name = input_name
        self.size = size
        self.i = 0

    def get_next(self):
        while self.i < len(self.paths):
            img = cv2.imread(self.paths[self.i])
            self.i += 1
            if img is None:
                continue
            x = letterbox(img, self.size)[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            return {self.input_name: np.ascontiguousarray(x)}
        return None


def head_nodes(model: onnx.ModelProto):
    idx = [int(m.group(1)) for n in model.graph.node for m in [re.match(r"/model\.(\d+)/", n.name)] if m]
    head = max(idx)
    branch = ("/cv2.", "/cv3.", "/cv4.")  # box / class / key-point conv branches: safe to quantise
    keep_fp32 = [n.name for n in model.graph.node
                 if n.name.startswith(f"/model.{head}/") and not (n.op_type == "Conv" and any(b in n.name for b in branch))]
    # decode (DFL, anchors, concat, sigmoid, ...) and anything outside the module tree stays FP32
    keep_fp32 += [n.name for n in model.graph.node if not n.name.startswith("/model.")]
    return keep_fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", default=None, help="folder of calibration images")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.model.replace(".onnx", ".int8.onnx")
    pre = args.model.replace(".onnx", ".pre.onnx")
    quant_pre_process(args.model, pre, skip_symbolic_shape=True)
    m = onnx.load(pre)
    inp = m.graph.input[0]
    size = inp.type.tensor_type.shape.dim[2].dim_value or 640
    paths = []
    if args.calib:
        paths = sorted(glob.glob(os.path.join(args.calib, "*")))
    if not paths:  # generic frames: demo reel
        cap = cv2.VideoCapture("demo/demo_campus.mp4")
        os.makedirs("data/calib_frames", exist_ok=True)
        k = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if k % 5 == 0:
                p = f"data/calib_frames/{k:05d}.jpg"
                cv2.imwrite(p, f)
                paths.append(p)
            k += 1
    random.Random(0).shuffle(paths)
    exclude = head_nodes(m)
    print(f"calibrating on {min(args.n, len(paths))} images, keeping {len(exclude)} head nodes in FP32")
    quantize_static(pre, out, Reader(paths[:args.n], inp.name, size), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    reduce_range=True, calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=exclude,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    os.remove(pre)
    print("wrote", out, os.path.getsize(out) // 1024, "KB (fp32:", os.path.getsize(args.model) // 1024, "KB)")


if __name__ == "__main__":
    main()
