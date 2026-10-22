"""ONNX Runtime wrapper for Ultralytics-style YOLO detectors (detect + pose).

Runs fully on CPU with no PyTorch dependency, which is what an edge box
(Raspberry Pi 5 / Jetson CPU / NUC) would run. Supports:

* plain detection heads  -> output (1, 4 + nc, N)
* pose heads (COCO-17)   -> output (1, 4 + 1 + 17*3, N)
* an optional extra "feature" output (an intermediate neck tensor) that
  the XAI layer uses for gradient-free Eigen-CAM saliency.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

try:  # onnxruntime is the only hard runtime dependency for inference
    import onnxruntime as ort
except ImportError:  # pragma: no cover - surfaced at construction time
    ort = None


@dataclass
class Detection:
    """A single detection in original-image pixel coordinates."""

    box: np.ndarray  # (4,) x1, y1, x2, y2
    score: float
    cls: int
    label: str
    keypoints: Optional[np.ndarray] = None  # (17, 3) x, y, conf for pose models
    track_id: int = -1

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def wh(self) -> np.ndarray:
        return np.array([self.box[2] - self.box[0], self.box[3] - self.box[1]])


@dataclass
class DetectorOutput:
    detections: List[Detection]
    feature: Optional[np.ndarray]  # (C, h, w) intermediate feature map if exported
    letterbox: tuple  # (scale, pad_x, pad_y, in_w, in_h)
    latency_ms: float
    extra: dict = field(default_factory=dict)


def letterbox(img: np.ndarray, size: int, color=(114, 114, 114)):
    """Resize keeping aspect ratio and pad to a square ``size`` input."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, scale, pad_x, pad_y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Plain greedy NMS (boxes in xyxy). Returns kept indices."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


class OnnxYolo:
    """Minimal, dependency-light YOLO inference engine."""

    def __init__(
        self,
        model_path: str,
        class_names: Sequence[str],
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        pose: bool = False,
        threads: Optional[int] = None,
        class_agnostic_nms: bool = False,
    ) -> None:
        if ort is None:
            raise ImportError("onnxruntime is required: pip install onnxruntime")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found. Run `python scripts/setup_models.py` first.")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2))
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # Respect a fixed export size if the model has one.
        if isinstance(shape[2], int):
            imgsz = shape[2]
        self.imgsz = imgsz
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.class_names = list(class_names)
        self.nc = len(class_names)
        self.conf = conf
        self.iou = iou
        self.pose = pose
        self.class_agnostic_nms = class_agnostic_nms
        self.model_path = model_path

    # ------------------------------------------------------------------ utils
    def preprocess(self, img_bgr: np.ndarray):
        lb, scale, px, py = letterbox(img_bgr, self.imgsz)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), (scale, px, py, self.imgsz, self.imgsz)

    def _postprocess(self, pred: np.ndarray, lbinfo, orig_shape) -> List[Detection]:
        scale, px, py, _, _ = lbinfo
        p = pred[0].T  # (N, 4+...)
        boxes_xywh = p[:, :4]
        if self.pose:
            scores = p[:, 4]
            cls = np.zeros(len(p), dtype=int)
            kpts = p[:, 5:5 + 51].reshape(-1, 17, 3)
        else:
            cls_scores = p[:, 4:4 + self.nc]
            cls = cls_scores.argmax(1)
            scores = cls_scores[np.arange(len(p)), cls]
            kpts = None
        m = scores >= self.conf
        if not m.any():
            return []
        boxes_xywh, scores, cls = boxes_xywh[m], scores[m], cls[m]
        if kpts is not None:
            kpts = kpts[m]
        xyxy = np.empty_like(boxes_xywh)
        xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        if self.class_agnostic_nms:
            keep = nms(xyxy, scores, self.iou)
        else:  # offset trick for per-class NMS
            off = cls[:, None].astype(np.float32) * 4096.0
            keep = nms(xyxy + off, scores, self.iou)
        h0, w0 = orig_shape[:2]
        dets = []
        for i in keep:
            b = xyxy[i].copy()
            b[[0, 2]] = ((b[[0, 2]] - px) / scale).clip(0, w0)
            b[[1, 3]] = ((b[[1, 3]] - py) / scale).clip(0, h0)
            kp = None
            if kpts is not None:
                kp = kpts[i].copy()
                kp[:, 0] = (kp[:, 0] - px) / scale
                kp[:, 1] = (kp[:, 1] - py) / scale
            c = int(cls[i])
            label = self.class_names[c] if c < len(self.class_names) else str(c)
            dets.append(Detection(box=b, score=float(scores[i]), cls=c, label=label, keypoints=kp))
        return dets

    # --------------------------------------------------------------- inference
    def __call__(self, img_bgr: np.ndarray) -> DetectorOutput:
        t0 = time.perf_counter()
        x, lbinfo = self.preprocess(img_bgr)
        outs = self.session.run(self.outputs, {self.input_name: x})
        dets = self._postprocess(outs[0], lbinfo, img_bgr.shape)
        feature = None
        if len(outs) > 1:
            f = outs[-1]
            feature = f[0] if f.ndim == 4 else None
        return DetectorOutput(dets, feature, lbinfo, (time.perf_counter() - t0) * 1e3)
