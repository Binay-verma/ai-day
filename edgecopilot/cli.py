"""Headless runner: process a video/camera, print explained alerts, optionally save output.

    python -m edgecopilot.cli --source demo/demo_campus.mp4 --zone "Corridor B" --save outputs/annotated.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from .copilot.events import EventLog
from .io.sensors import SensorHub
from .io.video import VideoSource
from .pipeline import Copilot, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Explainable Edge AI Copilot - headless run")
    ap.add_argument("--source", default=os.path.join(ROOT, "demo", "demo_campus.mp4"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "campus.yaml"))
    ap.add_argument("--zone", default="Corridor B")
    ap.add_argument("--sensors", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--save", default=None, help="write an annotated, privacy-filtered video")
    ap.add_argument("--events", default=os.path.join(ROOT, "logs", "events.jsonl"))
    ap.add_argument("--json", action="store_true", help="print alerts as JSON lines")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cop = Copilot(cfg, root=ROOT, zone=args.zone, sensors=SensorHub(fallback_zone="default"), event_log=EventLog(args.events))
    csv_path = args.sensors or (os.path.splitext(str(args.source))[0] + "_sensors.csv")
    if os.path.exists(csv_path):
        cop.sensors.load_csv(csv_path)
    src = VideoSource(args.source, stride=args.stride, max_frames=args.max_frames)
    writer = None
    lat = []
    t_wall = time.perf_counter()
    n = 0
    for frame, t in src:
        cop.sensors.advance(t)
        res = cop.process(frame, t)
        lat.append(res.timings["total"])
        n += 1
        if res.scene_change and not args.json:
            print(f"[{t:6.2f}s] scene change detected - baselines reset")
        for al in res.new_alerts:
            if args.json:
                print(json.dumps(al.summary(), default=float))
            else:
                print(f"[{t:6.2f}s] {al.severity:8s} {al.triage.get('mode', '')}: {al.explanation['text']}")
                print(f"          grounded {al.verification['groundedness']:.0%} of {al.verification['n_claims']} claims | "
                      f"decided in {al.processing_ms + al.explain_ms:.0f} ms")
        if args.save:
            img = cop.render(cop._last_frame, res.people, res.hazards, res.assessments)
            if writer is None:
                os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter.fourcc(*"mp4v"), src.fps / args.stride,
                                         (img.shape[1], img.shape[0]))
            writer.write(img)
    if writer:
        writer.release()
    wall = time.perf_counter() - t_wall
    if not args.json:
        print(f"\n{n} frames in {wall:.1f}s ({n / max(wall, 1e-9):.1f} FPS), latency p50 {np.median(lat):.1f} ms, "
              f"p95 {np.percentile(lat, 95):.1f} ms, scene changes {cop.scene_changes}")


if __name__ == "__main__":
    main()
