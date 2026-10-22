"""System-level fire benchmark on the FURG fire video dataset (23 videos).

Unlike the image benchmark this runs the *whole* fire branch of the copilot
(detector every N frames + dark-channel/flicker cues + additive fusion +
hysteresis) on video, which is what an operator experiences:

* fire videos (17): detected?  time-to-alert after the first annotated flame
* non-fire videos (6, incl. a cleaning robot and patrol robot footage):
  false alarms, reported per hour of video
* explanation grounding of every alert raised

    python benchmarks/eval_fire_video.py --model models/fire_smoke.onnx --stride 3
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgecopilot.io.video import VideoSource  # noqa: E402
from edgecopilot.pipeline import Copilot, load_config  # noqa: E402

ROOT = "/home/claude/datasets/furg_fire"


def first_flame_frame(xml_path):
    s = open(xml_path).read()
    frames = re.findall(r"<frameNumber>(\d+)</frameNumber>\s*<annotations>(.*?)</annotations>", s, re.S)
    ann = [int(f) for f, a in frames if a.strip()]
    return (min(ann) if ann else None), len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/campus.yaml")
    ap.add_argument("--model", default=None, help="override fire/smoke ONNX path")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--out", default="benchmarks/results/fire_furg.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["models"]["fire_smoke"]["path"] = args.model
    rows = []
    for mp4 in sorted(glob.glob(os.path.join(ROOT, "*.mp4"))):
        name = os.path.splitext(os.path.basename(mp4))[0]
        xml = os.path.join(ROOT, name + ".xml")
        if not os.path.exists(xml):
            xml = next((x for x in glob.glob(os.path.join(ROOT, "*.xml")) if os.path.basename(x).lower() == name.lower() + ".xml"), None)
        onset_f, _ = first_flame_frame(xml) if xml else (None, 0)
        c = copy.deepcopy(cfg)
        cop = Copilot(c, root=".", zone="Lab Block A", enable={"fall": False, "crowd": False, "fire_smoke": True})
        src = VideoSource(mp4, stride=args.stride, max_frames=int(args.max_seconds * 30 / args.stride))
        onset_t = None if onset_f is None else onset_f / src.fps
        alerts, t_last, t0 = [], 0.0, time.perf_counter()
        for frame, t in src:
            res = cop.process(frame, t)
            for al in res.new_alerts:
                alerts.append({"t": round(t, 2), "prob": round(al.prob, 3), "triage": al.triage.get("mode"),
                               "grounded": al.verification["groundedness"], "because": al.explanation["because"]})
            t_last = t
        label = onset_f is not None
        first = alerts[0]["t"] if alerts else None
        rows.append({"video": name, "fire": label, "seconds": round(t_last, 1), "n_alerts": len(alerts),
                     "detected": bool(alerts) if label else None,
                     "time_to_alert_s": (round(first - onset_t, 2) if (label and first is not None) else None),
                     "false_alarms": (len(alerts) if not label else sum(1 for a in alerts if onset_t is not None and a["t"] < onset_t)),
                     "alerts": alerts[:3], "proc_s": round(time.perf_counter() - t0, 1)})
        r = rows[-1]
        print(f"{name:28s} fire={label!s:5} alerts={len(alerts)} tta={r['time_to_alert_s']} fa={r['false_alarms']} ({r['proc_s']}s)", flush=True)
    fire = [r for r in rows if r["fire"]]
    neg = [r for r in rows if not r["fire"]]
    tta = sorted(r["time_to_alert_s"] for r in fire if r["time_to_alert_s"] is not None)
    neg_hours = sum(r["seconds"] for r in neg) / 3600
    summary = {
        "fire_videos": len(fire), "detected": sum(bool(r["detected"]) for r in fire),
        "video_recall": sum(bool(r["detected"]) for r in fire) / max(len(fire), 1),
        "time_to_alert_median_s": tta[len(tta) // 2] if tta else None,
        "nonfire_videos": len(neg), "nonfire_minutes": round(neg_hours * 60, 1),
        "false_alarms_nonfire": sum(r["false_alarms"] for r in neg),
        "false_alarms_per_hour": sum(r["false_alarms"] for r in neg) / max(neg_hours, 1e-9),
        "all_alerts_grounded": all(a["grounded"] == 1.0 for r in rows for a in r["alerts"]),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "videos": rows, "stride": args.stride, "max_seconds": args.max_seconds}, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
