"""Edge latency / throughput benchmark of the full copilot pipeline.

Runs the production :class:`Copilot` on a video and reports per-stage
latency (p50/p95), processed FPS, the share of frames where the motion gate
skipped the heavy models, an energy proxy (CPU-seconds per minute of video)
and the alert time-to-decision.  Several configurations are compared:

* pose input 640 / 480 / 320
* FP32 vs INT8 ONNX models (if ``models/*.int8.onnx`` exist)
* motion-gated cascade on / off

    python benchmarks/bench_latency.py --video demo/demo_campus.mp4 --frames 600
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgecopilot.io.sensors import SensorHub  # noqa: E402
from edgecopilot.io.video import VideoSource  # noqa: E402
from edgecopilot.pipeline import Copilot, load_config  # noqa: E402


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def run(cfg, video, frames, zone, sensor_csv=None, stride=1):
    cop = Copilot(cfg, root=".", zone=zone, sensors=SensorHub(fallback_zone="default"))
    if sensor_csv and os.path.exists(sensor_csv):
        cop.sensors.load_csv(sensor_csv)
    stages, gated, alerts = {}, 0, []
    n = 0
    c0, w0 = cpu_seconds(), time.perf_counter()
    t_video = 0.0
    for frame, t in VideoSource(video, stride=stride, max_frames=frames):
        cop.sensors.advance(t)
        res = cop.process(frame, t)
        for k, v in res.timings.items():
            stages.setdefault(k, []).append(v)
        gated += int(not res.ran_detector)
        for a in res.new_alerts:
            alerts.append({"hazard": a.hazard, "t": round(t, 2), "decide_ms": round(a.processing_ms + a.explain_ms, 1),
                           "triage": a.triage.get("mode")})
        n += 1
        t_video = t
    wall = time.perf_counter() - w0
    cpu = cpu_seconds() - c0
    return {
        "frames": n,
        "fps_processed": n / wall,
        "stages_ms": {k: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(np.mean(v))}
                      for k, v in stages.items()},
        "gated_fraction": gated / max(n, 1),
        "cpu_s_per_video_min": cpu / max(t_video / 60, 1e-9),
        "video_seconds": t_video,
        "alerts": alerts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="demo/demo_campus.mp4")
    ap.add_argument("--config", default="configs/campus.yaml")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--zone", default="Corridor B")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="benchmarks/results/latency.json")
    ap.add_argument("--quick", action="store_true", help="only the default configuration")
    args = ap.parse_args()
    base = load_config(args.config)
    sensor_csv = os.path.splitext(args.video)[0] + "_sensors.csv"
    variants = {"fp32_640": base}
    if not args.quick:
        def with_models(pose, fire, **rt):
            c = copy.deepcopy(base)
            c["models"]["pose"]["path"], c["models"]["fire_smoke"]["path"] = pose, fire
            c["runtime"].update(rt)
            return c
        if os.path.exists("models/yolo11n-pose.int8.onnx") and os.path.exists("models/fire_smoke.int8.onnx"):
            variants["int8_640"] = with_models("models/yolo11n-pose.int8.onnx", "models/fire_smoke.int8.onnx")
        if os.path.exists("models/yolo11n-pose_416.onnx") and os.path.exists("models/fire_smoke_416.onnx"):
            variants["fp32_416"] = with_models("models/yolo11n-pose_416.onnx", "models/fire_smoke_416.onnx")
        variants["fp32_640_no_cascade"] = with_models(base["models"]["pose"]["path"], base["models"]["fire_smoke"]["path"],
                                                      motion_gate=False, fire_every=1)
    res = {"cpu_count": os.cpu_count(), "threads": base["runtime"]["threads"], "video": args.video, "variants": {}}
    for name, cfg in variants.items():
        r = run(cfg, args.video, args.frames, args.zone, sensor_csv, args.stride)
        res["variants"][name] = r
        print(f"{name:16s} fps={r['fps_processed']:.1f} total p50={r['stages_ms']['total']['p50']:.1f} "
              f"p95={r['stages_ms']['total']['p95']:.1f} gated={r['gated_fraction']:.2f} cpu_s/min={r['cpu_s_per_video_min']:.1f} "
              f"alerts={[(a['hazard'], a['t']) for a in r['alerts']]}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
