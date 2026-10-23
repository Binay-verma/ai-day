"""Visual evidence maps for alerts.

* :func:`eigencam` - gradient-free Eigen-CAM (Muhammad & Yeasin, IJCNN 2020)
  from an intermediate feature map exported as an extra ONNX output.  It
  needs no back-propagation, so it runs on the same INT8/FP32 ONNX model on
  the edge device (one small SVD, ~1-3 ms).
* :func:`box_renormalize` - re-normalise a CAM inside each detection box so
  every alerted object gets its own visible evidence (as in pytorch-grad-cam's
  ``renormalize_cam_in_bounding_boxes``).
* :func:`occlusion_saliency` - model-agnostic perturbation saliency used as
  an additional reference in benchmarks.
* :func:`overlay` - draw a heat-map on the frame for the dashboard.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np


def eigencam(feature: np.ndarray, lbinfo, orig_shape) -> np.ndarray:
    """Project (C, h, w) activations on their first principal component.

    Returns a [0, 1] map in original-image coordinates (letterbox undone).
    """
    C, h, w = feature.shape
    A = feature.reshape(C, h * w).T.astype(np.float32)  # (hw, C)
    A = A - A.mean(axis=0, keepdims=True)
    # first right-singular vector via the small C x C covariance (cheap: C<=256)
    cov = A.T @ A
    vals, vecs = np.linalg.eigh(cov)
    v1 = vecs[:, -1]
    cam = (A @ v1).reshape(h, w)
    # sign of a principal component is arbitrary: orient towards high activation energy
    energy = np.abs(feature).sum(0)
    if np.corrcoef(cam.ravel(), energy.ravel())[0, 1] < 0:
        cam = -cam
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-9)
    return _unletterbox(cam, lbinfo, orig_shape)


def _unletterbox(cam: np.ndarray, lbinfo, orig_shape) -> np.ndarray:
    scale, px, py, in_w, in_h = lbinfo
    up = cv2.resize(cam, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    h0, w0 = orig_shape[:2]
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
    crop = up[py:py + nh, px:px + nw]
    if crop.size == 0:
        return np.zeros((h0, w0), np.float32)
    return cv2.resize(crop, (w0, h0), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def box_renormalize(cam: np.ndarray, boxes: Sequence[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(cam)
    h, w = cam.shape
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = cam[y1:y2, x1:x2]
        rng = patch.max() - patch.min()
        norm = (patch - patch.min()) / (rng + 1e-9) if rng > 1e-6 else np.ones_like(patch) * 0.5
        out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], norm)
    return out


def occlusion_saliency(img: np.ndarray, score_fn: Callable[[np.ndarray], float], grid: int = 8,
                       fill: Optional[Sequence[int]] = None) -> np.ndarray:
    """Score drop when each grid cell is greyed out (model agnostic, slow)."""
    h, w = img.shape[:2]
    base = score_fn(img)
    sal = np.zeros((grid, grid), np.float32)
    fill = fill if fill is not None else (114, 114, 114)
    for i in range(grid):
        for j in range(grid):
            im = img.copy()
            im[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid] = fill
            sal[i, j] = max(0.0, base - score_fn(im))
    out = np.asarray(cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)
    return out / (out.max() + 1e-9)


def overlay(img: np.ndarray, sal: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((np.clip(sal, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if heat.shape[:2] != img.shape[:2]:
        heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
    mask = (sal > 0.05)[..., None] if sal.shape == img.shape[:2] else True
    blended = cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
    return np.where(mask, blended, img)


def skeleton_saliency(shape, keypoints_list: List[np.ndarray], conf: float = 0.3) -> np.ndarray:
    """Evidence map for pose-based alerts: the torso/limb key-points used."""
    h, w = shape[:2]
    sal = np.zeros((h, w), np.float32)
    for kp in keypoints_list:
        for x, y, c in kp:
            if c >= conf and 0 <= x < w and 0 <= y < h:
                cv2.circle(sal, (int(x), int(y)), max(4, h // 60), float(c), -1)
    sal = cv2.GaussianBlur(sal, (0, 0), max(2, h // 120))
    return sal / (sal.max() + 1e-9)
