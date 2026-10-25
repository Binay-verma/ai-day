"""Exit-edge prediction and exit counting on UMN (11 clips, 3 scenes, sudden dispersals).

Ground truth for prediction is self-supervised: the edge through which each track actually left
the view (``ExitMonitor`` exit events). The prediction for a track is its heading ``h`` seconds
before it left. Baselines: the edge nearest to the person at that time, and the clip's most common
exit edge (an oracle that knows the answer distribution).

Ground truth for counting is a hand count of the people visible at dispersal onset in the five
outdoor clips, where every way out is an image edge (lawn: clips 0-1, plaza: 8-10; counted by eye
on the onset frame, +-1 person). Indoor clips (people run away down the hall and shrink into the
depth) are reported separately: those people mostly vanish inside the image, not at an edge.

    python benchmarks/eval_exits.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgecopilot.perception.detector import Detection  # noqa: E402
from edgecopilot.perception.exits import EDGES, ExitMonitor, predict_edge  # noqa: E402
from edgecopilot.perception.tracker import Tracker  # noqa: E402

ROOT_DS = os.environ.get("EDGECOPILOT_DATASETS", "/home/claude/datasets")
HAND_COUNT = {0: 15, 1: 14, 8: 12, 9: 6, 10: 7}   # people visible at onset who then leave (outdoor clips)
OUTDOOR = set(HAND_COUNT)
FPS = 30.0
W, H = 320, 240
BANNER_Y = 0.14  # burnt-in banner covers the top-left; detections there are not people


def nearest_edge(box, w=W, h=H):
    c = (box[:2] + box[2:]) / 2
    d = {"left": c[0], "right": w - c[0], "top": c[1], "bottom": h - c[1]}
    return min(d, key=d.get)


def score_exits(frames_dets, shape, horizons, acc, skip=None):
    """Replay tracker + ExitMonitor; score predict_edge vs nearest edge at each horizon before every edge exit."""
    hh, ww = shape
    tr, mon = Tracker(high_thr=0.35), ExitMonitor()
    hist, events = defaultdict(list), []
    for t, dets in frames_dets:
        tr.update([Detection(np.asarray(b, float), float(s), 0, "person", None) for b, s in dets], t)
        live = dict(tr.tracks)
        for i, x in live.items():
            if x.misses == 0:
                hist[i].append((t, np.asarray(x.last.box, float), np.asarray(x.velocity(0.5), float)))
        events += mon.update(live, (hh, ww))
    edge_events = [e for e in events if e.edge is not None and not (skip and skip(e))]
    for e in edge_events:
        rows = hist[e.track_id]
        for h in horizons:
            past = [r for r in rows if r[0] <= e.t - h]
            if not past or rows[0][0] > e.t - h - 0.25:
                continue
            box, v = past[-1][1], past[-1][2]
            c = acc[h]
            c["n"] += 1
            c["predict_ok"] += int(predict_edge(box, v, (hh, ww))[0] == e.edge)
            c["nearest_ok"] += int(nearest_edge(box, ww, hh) == e.edge)
    return events, edge_events, hist


def summarise(acc):
    out = {}
    for h, c in acc.items():
        n = c["n"]
        out[str(h)] = {"exits_scored": n, "predicted_accuracy": c["predict_ok"] / n if n else None,
                       "predicted_ci95": wilson(c["predict_ok"], n),
                       "nearest_edge_accuracy": c["nearest_ok"] / n if n else None, "nearest_ci95": wilson(c["nearest_ok"], n)}
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def run_clip(frames, people, t0):
    tr, mon = Tracker(high_thr=0.35), ExitMonitor()
    heads = defaultdict(list)       # tid -> [(t, edge|None, eta, box)]
    events = []
    for idx, t in frames:
        dets = [Detection(np.asarray(b, float), float(s), 0, "person", None) for b, s in people.get(idx, [])]
        tr.update(dets, t - t0)
        live = {i: x for i, x in tr.tracks.items()}
        hd = {h.track_id: h for h in mon.headings(live, (H, W))}
        for i, x in live.items():
            if x.misses == 0:
                h = hd.get(i)
                heads[i].append((t - t0, h.edge if h else None, h.eta_s if h else None, np.asarray(x.last.box, float)))
        events += mon.update(live, (H, W))
    return events, heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/umn_people.pkl")
    ap.add_argument("--walk-cache", default="data/walk_people.pkl")
    ap.add_argument("--out", default="benchmarks/results/exits.json")
    a = ap.parse_args()
    people = pickle.load(open(a.cache, "rb"))
    lab = pd.read_csv(os.path.join(ROOT_DS, "umn_crowd", "umn_labels.csv"))
    res = {"dataset": "UMN (11 clips)", "horizons": {}, "counting": [], "dispersal_direction": []}
    pred = {h: Counter() for h in (0.5, 1.0, 2.0)}
    for clip, d in lab.groupby("clip_id"):
        frames = [(int(f), f / FPS) for f in d.frame]
        t0 = frames[0][1]
        onset = (d[d.abnormal == 1].frame.min()) / FPS - t0
        events, heads = run_clip(frames, people, t0)
        edge_events = [e for e in events if e.edge is not None and not (e.edge == "top" and e.box[0] < 0.56 * W and e.box[1] < BANNER_Y * H)]
        majority = Counter(e.edge for e in edge_events).most_common(1)[0][0] if edge_events else None
        for e in edge_events:
            hist = heads.get(e.track_id, [])
            for h in pred:
                past = [r for r in hist if r[0] <= e.t - h]
                if not past or hist[0][0] > e.t - h - 0.25:
                    continue
                r = past[-1]
                c = pred[h]
                c["n"] += 1
                c["heading_ok"] += int(r[1] == e.edge)
                c["heading_none"] += int(r[1] is None)
                c["nearest_ok"] += int(nearest_edge(r[3]) == e.edge)
                c["majority_ok"] += int(majority == e.edge)
        after = [e for e in edge_events if e.t >= onset - 0.2]
        vanished = [e for e in events if e.edge is None and e.t >= onset - 0.2]
        row = {"clip": int(clip), "outdoor": clip in OUTDOOR, "exits_at_edges_after_onset": len(after),
               "vanished_inside_after_onset": len(vanished), "hand_count": HAND_COUNT.get(int(clip))}
        res["counting"].append(row)
        # dispersal direction: predicted share per edge in the first second after onset vs actual exit share
        early = Counter()
        for tid, hist in heads.items():
            r = [x for x in hist if onset <= x[0] <= onset + 1.0 and x[1] is not None]
            if r:
                early[r[-1][1]] += 1
        actual = Counter(e.edge for e in after)
        if early and actual:
            res["dispersal_direction"].append({"clip": int(clip), "predicted_top_edge": early.most_common(1)[0][0],
                                               "actual_top_edge": actual.most_common(1)[0][0],
                                               "predicted": dict(early), "actual": dict(actual)})
    for h, c in pred.items():
        n = c["n"]
        res["horizons"][str(h)] = {
            "exits_scored": n,
            "heading_accuracy": c["heading_ok"] / n if n else None, "heading_ci95": wilson(c["heading_ok"], n),
            "no_heading_share": c["heading_none"] / n if n else None,
            "nearest_edge_accuracy": c["nearest_ok"] / n if n else None, "nearest_ci95": wilson(c["nearest_ok"], n),
            "clip_majority_oracle_accuracy": c["majority_ok"] / n if n else None}
    out = [r for r in res["counting"] if r["outdoor"]]
    err = [r["exits_at_edges_after_onset"] - r["hand_count"] for r in out]
    res["counting_outdoor_summary"] = {"clips": len(out), "hand_total": sum(r["hand_count"] for r in out),
                                       "counted_total": sum(r["exits_at_edges_after_onset"] for r in out),
                                       "mean_abs_error_people": float(np.mean(np.abs(err))),
                                       "mean_abs_pct_error": float(np.mean([abs(e) / r["hand_count"] for e, r in zip(err, out)]))}
    dd = res["dispersal_direction"]
    res["dispersal_direction_summary"] = {"clips": len(dd), "top_edge_agree": sum(x["predicted_top_edge"] == x["actual_top_edge"] for x in dd)}
    # held-out walking footage (people walking steadily in or out of view; detections from extract_walk_dets.py)
    horizons = (0.5, 1.0, 2.0, 3.0)
    acc_umn = {h: Counter() for h in horizons}
    for clip, d in lab.groupby("clip_id"):
        fr = [(f / FPS, people.get(int(f), [])) for f in d.frame]
        score_exits(fr, (H, W), horizons, acc_umn,
                    skip=lambda e: e.edge == "top" and e.box[0] < 0.56 * W and e.box[1] < BANNER_Y * H)
    res["prediction_umn"] = summarise(acc_umn)
    if os.path.exists(a.walk_cache):
        walk = pickle.load(open(a.walk_cache, "rb"))
        acc_w = {h: Counter() for h in horizons}
        per = {}
        for name, c in walk.items():
            ev, ed, _ = score_exits(c["frames"], tuple(c["shape"]), horizons, acc_w)
            per[name] = {"edge_exits": len(ed), "vanished_inside": len(ev) - len(ed)}
        res["prediction_walking"] = summarise(acc_w)
        res["walking_clips"] = per
        acc_all = {h: acc_umn[h] + acc_w[h] for h in horizons}
        res["prediction_pooled"] = summarise(acc_all)
    print(json.dumps({k: v for k, v in res.items() if k not in ("counting", "dispersal_direction", "horizons")}, indent=1))
    print(json.dumps(res["counting"], indent=0))
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    _ = EDGES
    main()
