"""External sanity check on footage never used during development.

Every other benchmark in this repository was also used while the cues were
being designed (see docs/RESULTS.md caveats).  This script runs the frozen
system - the shipped ``configs/campus.yaml`` and models, with no parameter
changed after the first run - on clips from sources that were never looked at
during development:

* falls: CAUCAFall sample (5 clips, Eraso et al., CC BY 4.0), UR-Fall samples
  (2 falls, Kwolek & Kepski), and 4 web fall examples shipped with an
  open-source fall-detection repo (``SyedBurhanAhmed/Real-Time-Fall-Detection-using-YOLO``);
* normal activity: UR-Fall ADL sample plus 6 Intel IoT DevKit sample videos
  (walking, aisles, classroom, worker zone) - all-hazard false alarms per hour.

UR-Fall cam-0 videos store depth (left) and RGB (right) side by side; only the RGB half
is used. (The first run of this script fed the whole double frame - an input bug, corrected
on 2026-10-24 and disclosed in docs/RESULTS.md.)

Many fall clips end within a second of the impact (UR-Fall: 2.0-2.5 s total),
so each fall clip is followed by the last frame held for ``--hold`` seconds,
i.e. "the person stays down". This is stated in the output.

It is a *sanity check*, not a benchmark: 11 falls and ~7 minutes of normal
video give wide intervals, which are reported (Wilson / Poisson).

    python benchmarks/eval_external.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgecopilot.io.video import VideoSource  # noqa: E402
from edgecopilot.pipeline import Copilot, load_config  # noqa: E402

DS = "/home/claude/datasets"
FALLS = sorted(glob.glob(f"{DS}/fall_extras/cauca_sample/*.avi") + glob.glob(f"{DS}/fall_extras/urfd_samples/fall-*.mp4")
               + glob.glob(f"{DS}/fall_extras/punpayut_examples/*.mp4"))
NORMAL = [f"{DS}/fall_extras/urfd_samples/adl-10-cam0.mp4"] + [
    f"{DS}/test_videos/{n}" for n in ("classroom.mp4", "face-demographics-walking.mp4", "one-by-one-person-detection.mp4",
                                      "people-detection.mp4", "store-aisle-detection.mp4", "worker-zone-detection.mp4")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def poisson_ci(k, z=1.96):
    """Approximate 95 % interval for a Poisson count (Wilson-Hilferty / exact-ish)."""
    lo = 0.0 if k == 0 else k * (1 - 1 / (9 * k) - z / (3 * math.sqrt(k))) ** 3
    k1 = k + 1
    hi = k1 * (1 - 1 / (9 * k1) + z / (3 * math.sqrt(k1))) ** 3
    return lo, hi


def run(path, target_fps, hold_s, config="configs/campus.yaml"):
    cfg = load_config(config)
    cop = Copilot(cfg, root=".", zone="External test")
    src = VideoSource(path)
    stride = max(1, round(src.fps / target_fps))
    src = VideoSource(path, stride=stride)
    alerts, last, t = [], None, 0.0
    side_by_side = os.path.basename(path).startswith(("fall-", "adl-"))  # UR cam0 = depth | RGB
    for frame, t in src:
        if side_by_side and frame.shape[1] >= 2 * frame.shape[0]:
            frame = frame[:, frame.shape[1] // 2:]
        res = cop.process(frame, t)
        alerts += res.new_alerts
        last = frame
    dur = t
    if hold_s > 0 and last is not None:
        dt = stride / src.fps
        for i in range(1, int(hold_s / dt) + 1):
            res = cop.process(last, dur + i * dt)
            alerts += res.new_alerts
    return dur, [{"hazard": a.hazard, "t": round(a.t_video, 2), "prob": round(a.prob, 3), "mode": a.triage.get("mode", ""),
                  "grounded": not a.verification.get("unsupported"), "text": a.explanation.get("text", "")} for a in alerts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--hold", type=float, default=4.0, help="seconds the last frame of a fall clip is held")
    ap.add_argument("--out", default="benchmarks/results/external_sanity.json")
    ap.add_argument("--config", default="configs/campus.yaml")
    a = ap.parse_args()
    falls, normals = [], []
    for p in FALLS:
        dur, al = run(p, a.fps, a.hold, a.config)
        hit = any(x["hazard"] == "fall" for x in al)
        falls.append({"clip": os.path.relpath(p, DS), "seconds": round(dur, 1), "detected": hit, "alerts": al})
        print(f"FALL   {'HIT ' if hit else 'MISS'} {os.path.basename(p):28s} {[(x['hazard'], x['t'], x['mode']) for x in al]}")
    for p in NORMAL:
        dur, al = run(p, a.fps, 0, a.config)
        normals.append({"clip": os.path.relpath(p, DS), "seconds": round(dur, 1), "false_alarms": len(al), "alerts": al})
        print(f"NORMAL FA={len(al)} {os.path.basename(p):34s} {dur:6.1f}s {[(x['hazard'], x['t'], x['mode']) for x in al]}")
    k = sum(f["detected"] for f in falls)
    fa = sum(n["false_alarms"] for n in normals)
    hours = sum(n["seconds"] for n in normals) / 3600
    lo, hi = poisson_ci(fa)
    all_alerts = [x for f in falls for x in f["alerts"]] + [x for n in normals for x in n["alerts"]]
    summary = {
        "protocol": f"frozen system ({a.config} as shipped), sources never used in development; "
                    f"fall clips followed by the last frame held {a.hold} s; {a.fps} fps",
        "fall_recall": f"{k}/{len(falls)}", "fall_recall_wilson95": wilson(k, len(falls)),
        "normal_minutes": round(hours * 60, 1), "false_alarms_all_hazards": fa,
        "false_alarms_per_hour": round(fa / hours, 1) if hours else None,
        "false_alarms_per_hour_95ci": [round(lo / hours, 1), round(hi / hours, 1)] if hours else None,
        "alerts_grounded": f"{sum(1 for x in all_alerts if x['grounded'])}/{len(all_alerts)}",
    }
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "falls": falls, "normal": normals}, f, indent=1)


if __name__ == "__main__":
    main()
