"""Second fall dataset: UR Fall Detection (Kwolek & Kepski 2014), camera 0, frozen system.

Never used during development. 30 fall clips (GitHub mirror cwlroda/falldetection_openpifpaf)
and 5 daily-activity clips (chandrika-kantipudi/elderly-fall-surveillance; the other 35 UR
ADL clips are not mirrored on GitHub). UR cam-0 videos hold depth (left) and RGB (right) side
by side; only the RGB half is used. Per-frame labels from the official UR CSV (label 0 =
falling, 1 = lying) give the fall onset for time-to-detect.

The clips end about a second after the person reaches the floor, so the last frame is held
for ``--hold`` seconds ("the person stays down"), as in benchmarks/eval_external.py.

    python benchmarks/eval_urfall.py                                        # shipped config
    python benchmarks/eval_urfall.py --config configs/profiles/fall_accurate.yaml --out ...
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgecopilot.pipeline import Copilot, load_config  # noqa: E402

DS = os.environ.get("EDGECOPILOT_DATASETS", "/home/claude/datasets")
FALLS = sorted(glob.glob(f"{DS}/ext/urfall_cwlroda/test/input/fall-*-cam0.mp4"))
ADLS = sorted(glob.glob(f"{DS}/ext/urfall_adl/fall_detection/not_fall/adl-*-cam0.mp4"))
LABELS = f"{DS}/fall_extras/urfd_features/urfall-cam0-falls.csv"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def onsets():
    first = {}
    if os.path.exists(LABELS):
        for row in csv.reader(open(LABELS)):
            name, frame, lab = row[0], int(row[1]), int(row[2])
            if lab == 0 and name not in first:
                first[name] = (frame - 1) / 30.0
    return first


def ur_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        w = f.shape[1]
        yield i / fps, (f[:, w // 2:] if w >= 2 * f.shape[0] else f)  # RGB half of the depth|RGB pair
        i += 1


def run(cfg_path, path, stride, hold):
    cop = Copilot(load_config(cfg_path), root=".", zone="UR test room")
    alerts, last, t = [], None, 0.0
    for k, (t, f) in enumerate(ur_frames(path)):
        if k % stride:
            continue
        alerts += cop.process(f, t).new_alerts
        last = f
    if last is not None and hold > 0:
        dt = stride / 30.0
        for j in range(1, int(hold / dt) + 1):
            alerts += cop.process(last, t + j * dt).new_alerts
    return t, alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/campus.yaml")
    ap.add_argument("--stride", type=int, default=2, help="30 fps source -> 15 fps")
    ap.add_argument("--hold", type=float, default=4.0)
    ap.add_argument("--out", default="benchmarks/results/fall_urfall.json")
    a = ap.parse_args()
    on = onsets()
    rows = []
    for p in FALLS + ADLS:
        name = os.path.basename(p).replace("-cam0.mp4", "")
        is_fall = name.startswith("fall")
        dur, al = run(a.config, p, a.stride, a.hold)
        falls = [x for x in al if x.hazard == "fall"]
        t0 = on.get(name)
        delay = (falls[0].t_video - t0) if (falls and t0 is not None) else None
        rows.append({"clip": name, "fall": is_fall, "detected": bool(falls), "alerts": [(x.hazard, round(x.t_video, 2), x.triage.get("mode", "")) for x in al],
                     "onset_s": t0, "delay_s": None if delay is None else round(delay, 2), "seconds": round(dur, 1)})
        print(f"{name:8s} {'FALL' if is_fall else 'ADL '} {'HIT ' if falls else '----'} {rows[-1]['alerts']}", flush=True)
    fr = [r for r in rows if r["fall"]]
    ad = [r for r in rows if not r["fall"]]
    k = sum(r["detected"] for r in fr)
    delays = sorted(r["delay_s"] for r in fr if r["delay_s"] is not None)
    fa = sum(len(r["alerts"]) for r in ad)
    summ = {"config": a.config, "fall_recall": f"{k}/{len(fr)}", "fall_recall_wilson95": wilson(k, len(fr)),
            "alarm_mode": sum(1 for r in fr for h, _, m in r["alerts"] if h == "fall" and m == "ALARM"),
            "median_delay_after_fall_onset_s": delays[len(delays) // 2] if delays else None,
            "adl_clips": len(ad), "adl_clips_with_any_alert": sum(1 for r in ad if r["alerts"]), "adl_alerts": fa,
            "protocol": f"frozen system, RGB half of UR cam0, {30 // a.stride} fps, last frame held {a.hold} s"}
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "clips": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
