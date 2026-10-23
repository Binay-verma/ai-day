"""Video sources: files, webcams and RTSP/HTTP streams through OpenCV."""
from __future__ import annotations

import time
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np


class VideoSource:
    """Iterates (frame_bgr, t_seconds).

    For files ``t`` is stream time (frame_idx / fps) so benchmarks are
    reproducible regardless of processing speed; for live sources it is wall
    time since start.  ``stride`` skips frames (e.g. 30 fps camera -> 10 fps).
    """

    def __init__(self, src: Union[str, int], stride: int = 1, loop: bool = False,
                 realtime: bool = False, max_frames: Optional[int] = None):
        self.src = int(src) if isinstance(src, str) and src.isdigit() else src
        self.stride = max(1, stride)
        self.loop = loop
        self.realtime = realtime
        self.max_frames = max_frames
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise OSError(f"cannot open video source {src!r}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 25.0
        self.live = isinstance(self.src, int) or str(self.src).startswith(("rtsp://", "http://", "https://"))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        idx = 0
        emitted = 0
        t_start = time.perf_counter()
        offset = 0.0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop and not self.live:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    offset += idx / self.fps
                    idx = 0
                    continue
                break
            if idx % self.stride == 0:
                t = (time.perf_counter() - t_start) if self.live else offset + idx / self.fps
                if self.realtime and not self.live:
                    lag = t - (time.perf_counter() - t_start)
                    if lag > 0:
                        time.sleep(lag)
                yield frame, t
                emitted += 1
                if self.max_frames and emitted >= self.max_frames:
                    break
            idx += 1
        self.cap.release()


def read_all(path: str, stride: int = 1, max_frames: Optional[int] = None):
    return list(VideoSource(path, stride=stride, max_frames=max_frames))
