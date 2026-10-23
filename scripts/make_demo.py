"""Build the demo reel (and its synchronised ESP32 sensor trace) from public datasets.

The reel simulates one operator console watching several campus cameras in
turn; each cut is detected by the copilot as a scene change (baselines reset).

  0  PETS-2009 / OpenCV vtest  - people walking on campus (normal)      -> no alert
  1  GMDCSA24 ADL              - lying down on a bed (hard negative)     -> no alert
  2  GMDCSA24 fall             - a real fall                             -> FALL
  3  FURG house1 (CC0)         - small fire starts in a model house      -> SMOKE/FIRE (+ MQ-2 rise)
  4  UMN lawn scene            - sudden crowd dispersal                  -> CROWD

The source videos keep their own licences, so the reel is generated locally and
is not committed to git:  python scripts/make_demo.py  ->  demo/demo_campus.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import cv2
import numpy as np

DS = "/home/claude/datasets"
W, H, FPS = 960, 540, 15.0

SEGMENTS = [
    {"name": "Campus walkway (normal)", "src": f"{DS}/test_videos/vtest.avi", "start": 0.0, "dur": 16.0},
    {"name": "Hostel room: lying down (normal)", "src": f"{DS}/gmdcsa24/Subject 1/ADL/01.mp4", "start": 0.0, "dur": 8.0},
    {"name": "Hostel room: fall", "src": f"{DS}/gmdcsa24/Subject 1/Fall/12.mp4", "start": 0.0, "dur": 6.8, "hold": 4.0},
    {"name": "Lab bench: fire", "src": f"{DS}/furg_fire/house1.mp4", "start": 0.0, "dur": 22.0, "fire_at": 112 / 29.0},
    {"name": "Main lawn: crowd panic", "src": f"{DS}/umn_crowd/Crowd-Activity-All.avi", "start": 250 / 30.0, "dur": 12.0, "crop_top": 0.15},
]


def letterbox(img, crop_top=0.0):
    if crop_top:
        img = img[int(img.shape[0] * crop_top):]
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    r = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.zeros((H, W, 3), np.uint8)
    y, x = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def read_segment(seg):
    cap = cv2.VideoCapture(seg["src"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    n_out = int(seg["dur"] * FPS)
    last = None
    for k in range(n_out):
        t = seg["start"] + k / FPS
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        if not ok:
            if last is None:
                break
            f = last
        last = f
        frames.append(letterbox(f, seg.get("crop_top", 0.0)))
    if seg.get("hold") and last is not None:  # keep the last frame (person stays down)
        frames += [frames[-1]] * int(seg["hold"] * FPS)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo/demo_campus.mp4")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    t = 0.0
    timeline, sensor_rows = [], []
    for seg in SEGMENTS:
        frames = read_segment(seg)
        t0 = t
        for f in frames:
            vw.write(f)
        t += len(frames) / FPS
        ev = {"segment": seg["name"], "start_s": round(t0, 2), "end_s": round(t, 2)}
        if "fire_at" in seg:
            ev["fire_starts_s"] = round(t0 + seg["fire_at"], 2)
        timeline.append(ev)
        print(ev)
    vw.release()
    # ESP32 sensor trace in the fire room (1 Hz): clean air, then the MQ-2 sees smoke
    fire = next(e for e in timeline if "fire_starts_s" in e)
    k = 0.0
    while k <= t:
        in_room = fire["start_s"] <= k <= fire["end_s"]
        ppm = 118 + 6 * np.sin(k / 5)
        temp = 26.5 + 0.1 * np.sin(k / 20)
        if in_room and k >= fire["fire_starts_s"] + 2.0:  # gas reaches the sensor ~2 s later
            d = k - fire["fire_starts_s"] - 2.0
            ppm += min(700.0, 45.0 * d)
            temp += min(6.0, 0.25 * d)
        sensor_rows.append({"zone": "default", "t": round(k, 1), "smoke_ppm": round(ppm, 1), "temp_c": round(temp, 2),
                            "sound_db": round(52 + 2 * np.sin(k), 1)})
        k += 1.0
    csv_path = os.path.splitext(args.out)[0] + "_sensors.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sensor_rows[0]))
        w.writeheader()
        w.writerows(sensor_rows)
    json.dump(timeline, open(os.path.splitext(args.out)[0] + "_timeline.json", "w"), indent=2)
    print("wrote", args.out, csv_path, f"{t:.1f}s")


if __name__ == "__main__":
    main()
