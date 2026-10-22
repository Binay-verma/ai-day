"""Export an Ultralytics .pt model to ONNX for the edge runtime.

Adds the neck feature map(s) feeding the detection head as extra graph
outputs so the edge runtime can compute Eigen-CAM without PyTorch.

    python scripts/export_onnx.py --weights models/yolo11n-pose.pt --out models/yolo11n-pose.onnx --feature p3
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

import onnx


def head_inputs(model: onnx.ModelProto):
    """Return the neck tensors (P3, P4, P5) consumed by the detection head."""
    nodes = model.graph.node
    idx = [int(m.group(1)) for n in nodes for m in [re.match(r"/model\.(\d+)/", n.name)] if m]
    head = max(idx)
    prefix = f"/model.{head}/"
    produced_in_head = {o for n in nodes if n.name.startswith(prefix) for o in n.output}
    feats = []
    for n in nodes:
        if n.name.startswith(prefix):
            for i in n.input:
                if i.startswith("/model.") and i not in produced_in_head and i not in feats and not i.startswith(prefix):
                    feats.append(i)
    # keep the order of the producing layer index (P3 < P4 < P5)
    feats.sort(key=lambda s: int(re.match(r"/model\.(\d+)/", s).group(1)))
    return feats


def add_feature_output(path_in: str, path_out: str, which: str = "p3") -> str:
    m = onnx.load(path_in)
    feats = head_inputs(m)
    if len(feats) < 3:
        raise RuntimeError(f"could not find 3 neck outputs, found {feats}")
    name = {"p3": feats[0], "p4": feats[1], "p5": feats[2]}[which]
    vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
    m.graph.output.append(vi)
    onnx.save(m, path_out)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--feature", choices=["none", "p3", "p4", "p5"], default="p3")
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    tmp = model.export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True, dynamic=False)
    if args.feature != "none":
        name = add_feature_output(tmp, args.out, args.feature)
        print(f"added feature output {name}")
    else:
        shutil.copy(tmp, args.out)
    print("wrote", args.out, os.path.getsize(args.out) // 1024, "KB")


if __name__ == "__main__":
    main()
