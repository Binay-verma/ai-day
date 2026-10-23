"""Privacy-by-design rendering.

The edge box never needs to ship raw video.  Three modes control what leaves
the device (dashboard stream, alert snapshots, event log):

* ``raw``      - full frame (only for on-premise debugging)
* ``blur``     - heads anonymised using the *pose key-points we already have*
                 (nose/eyes/ears), so no extra face detector is run
* ``skeleton`` - background blurred heavily, people reduced to skeletons;
                 identity is not recoverable but posture/falls are evident
"""
from __future__ import annotations

from typing import Iterable, List

import cv2
import numpy as np

HEAD = [0, 1, 2, 3, 4]
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def _head_box(kp: np.ndarray, box: np.ndarray, conf: float = 0.3):
    pts = np.array([kp[i, :2] for i in HEAD if kp[i, 2] >= conf])
    bw, bh = box[2] - box[0], box[3] - box[1]
    if len(pts) == 0:
        # no visible head key-points: take the top fifth of the person box
        return int(box[0]), int(box[1]), int(box[2]), int(box[1] + 0.22 * bh)
    cx, cy = pts.mean(0)
    r = max(0.18 * min(bw, bh) if bw < bh else 0.18 * bh, np.ptp(pts[:, 0]) if len(pts) > 1 else 0, 8)
    return int(cx - 1.4 * r), int(cy - 1.6 * r), int(cx + 1.4 * r), int(cy + 1.4 * r)


def anonymize(frame: np.ndarray, people: Iterable, mode: str = "blur", pose_available: bool = True) -> np.ndarray:
    """Privacy filter applied before any image is shown, stored or sent.

    ``pose_available=False`` means people cannot be located (no person model
    loaded): the filter then fails safe and blurs the whole frame instead of
    letting unblurred faces through.
    """
    if mode == "raw":
        return frame
    if not pose_available:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // 12), max(1, h // 12)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    out = frame.copy()
    h, w = out.shape[:2]
    people = list(people)
    if mode == "skeleton":
        out = cv2.GaussianBlur(cv2.resize(frame, (max(1, w // 16), max(1, h // 16))), (3, 3), 0)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        out = (out * 0.6).astype(np.uint8)
        for d in people:
            if d.keypoints is not None:
                draw_skeleton(out, d.keypoints)
        return out
    for d in people:
        if d.keypoints is None:
            x1, y1, x2, y2 = d.box.astype(int)
            hb = (x1, y1, x2, int(y1 + 0.22 * (y2 - y1)))
        else:
            hb = _head_box(d.keypoints, d.box)
        x1, y1, x2, y2 = max(0, hb[0]), max(0, hb[1]), min(w, hb[2]), min(h, hb[3])
        if x2 - x1 > 2 and y2 - y1 > 2:
            out[y1:y2, x1:x2] = _obscure(out[y1:y2, x1:x2])
    return out


BLOCKS = 4  # cells across the head: coarse enough that no face detector finds a face (benchmarks/eval_privacy.py)


def _obscure(roi: np.ndarray, blocks: int = BLOCKS) -> np.ndarray:
    """Coarse pixelation (``blocks`` cells across the head) with softened cell edges.

    A Gaussian blur leaves a face-shaped blob that an independent face detector still
    found in 14 % of cases; 4 x ~5 cells keep the head visible as a head but removes the
    eye/nose/mouth layout that face detection and recognition rely on.
    """
    hh, ww = roi.shape[:2]
    bx = max(1, blocks)
    by = max(1, int(round(blocks * hh / max(ww, 1))))
    cells = cv2.resize(roi, (bx, by), interpolation=cv2.INTER_AREA)
    up = cv2.resize(cells, (ww, hh), interpolation=cv2.INTER_NEAREST)
    k = max(3, (min(ww, hh) // 6) | 1)
    return cv2.GaussianBlur(up, (k, k), 0)


def draw_skeleton(img: np.ndarray, kp: np.ndarray, color=(80, 220, 255), conf: float = 0.3) -> None:
    for a, b in SKELETON:
        if kp[a, 2] >= conf and kp[b, 2] >= conf:
            cv2.line(img, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]), color, 2, cv2.LINE_AA)
    for x, y, c in kp:
        if c >= conf:
            cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)


def head_regions(people: Iterable) -> List[tuple]:
    return [_head_box(d.keypoints, d.box) for d in people if d.keypoints is not None]
