"""UMN crowd-panic benchmark (frame-level AUC + event-level detection delay).

The UMN video (11 clips, 3 scenes, 320x240 @ 30 fps) shows normal walking
followed by sudden dispersal.  Ground truth is the per-frame abnormal flag.
The burnt-in "Normal/Abnormal Crowd Activity" banner is masked so the model
cannot cheat by reading it.

Reports:
* frame AUC of every individual cue and of the fused crowd score
* per clip: detection delay after abnormal onset, false alarms before onset
* split protocol: weights re-fitted on scenes {1,2} and tested on scene 3 (and
  rotations) to show the fused model generalises to an unseen scene
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgecopilot.io.video import VideoSource  # noqa: E402
from edgecopilot.perception.crowd import CrowdFeatureExtractor  # noqa: E402
from edgecopilot.perception.detector import OnnxYolo  # noqa: E402
from edgecopilot.pipeline import load_config  # noqa: E402
from edgecopilot.xai.fusion import HazardModel, Hysteresis, fit_weights  # noqa: E402

BANNER = [0.0, 0.0, 0.56, 0.14]
# UMN clip -> scene (lawn: 0-1, indoor: 2-7, plaza: 8-10)
SCENE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3}


def auc(scores, y):
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s_sorted = scores[order]
    # average ranks for ties
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos, nneg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def extract(video, gt, meta, pose_cache, stride=1):
    cfg_people = pickle.load(open(pose_cache, "rb")) if os.path.exists(pose_cache) else None
    pose = None if cfg_people else OnnxYolo("models/yolo11n-pose.onnx", ["person"], conf=0.3, pose=True, threads=2)
    people = cfg_people or {}
    rows = []
    fx = None
    cur_clip = -1
    for idx, (frame, t) in enumerate(VideoSource(video)):
        clip = int(meta.clip_id.iloc[idx])
        if clip != cur_clip:
            fx = CrowdFeatureExtractor(ignore_regions=[BANNER])
            cur_clip = clip
            t0 = t
        if idx not in people:
            out = pose(frame)
            people[idx] = [(d.box, d.score) for d in out.detections]
        dets = people[idx]
        hs = [b[3] - b[1] for b, s in dets if b[3] - b[1] > 8]
        vals = fx(frame, t - t0, len(dets), capacity=40, person_height_px=float(np.median(hs)) if hs else None)
        rows.append({"frame": idx, "clip": clip, "scene": SCENE[clip], "t": t - t0, "y": int(gt.abnormal.iloc[idx]), **vals})
    if not cfg_people:
        pickle.dump(people, open(pose_cache, "wb"))
    return pd.DataFrame(rows)


def run_model(df, model, hyst_cfg):
    probs, active = [], []
    for clip, g in df.groupby("clip", sort=False):
        h = Hysteresis(**hyst_cfg)
        for _, r in g.iterrows():
            vals = {k: r[k] for k in model.specs if k in r and r[k] == r[k]}
            a = h.update("crowd", model.assess("zone", r.t, vals))
            probs.append(a.prob)
            active.append(a.active)
    return np.array(probs), np.array(active)


def events(df, active, fps=30.0):
    out = []
    for clip, g in df.groupby("clip", sort=False):
        idx = g.index.values
        y = g.y.values
        act = active[idx]
        onset = np.flatnonzero(y)[0] if y.any() else None
        pre = act[:onset] if onset is not None else act
        # count rising edges of alerts during the normal part = false alarms
        fa = int(np.sum(np.diff(np.concatenate([[0], pre.astype(int)])) == 1))
        det = None
        if onset is not None:
            hits = np.flatnonzero(act[onset:])
            det = hits[0] / fps if len(hits) else None
        out.append({"clip": int(clip), "false_alarms_before_onset": fa, "detected": det is not None,
                    "delay_s": None if det is None else round(det, 2), "normal_s": round(len(pre) / fps, 1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/claude/datasets/umn_crowd")
    ap.add_argument("--config", default="configs/campus.yaml")
    ap.add_argument("--out", default="benchmarks/results/crowd_umn.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cache = "data/umn_features.pkl"
    gt = pd.read_csv(os.path.join(args.root, "umn_groundtruth_redpx.csv"))
    meta = pd.read_csv(os.path.join(args.root, "umn_meta_jeevitha.csv"))
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
    else:
        df = extract(os.path.join(args.root, "Crowd-Activity-All.avi"), gt, meta, "data/umn_people.pkl")
        df.to_pickle(cache)
    cfg = load_config(args.config)
    model = HazardModel.from_config("crowd", cfg["hazards"]["crowd"])
    feats = [k for k in model.specs if k in df.columns]
    res = {"n_frames": len(df), "abnormal_frac": float(df.y.mean()),
           "feature_auc": {k: auc(df[k].fillna(0).values, df.y.values) for k in feats + ["speed", "energy", "motion_area", "divergence"] if k in df}}
    probs, active = run_model(df, model, cfg["hysteresis"])
    res["expert_model"] = {"frame_auc": auc(probs, df.y.values), "events": events(df, active)}
    # leave-one-scene-out refit of the additive weights
    specs = [model.specs[k] for k in feats]
    loso = {}
    all_p = np.zeros(len(df))
    for scene in sorted(df.scene.unique()):
        tr, te = df[df.scene != scene], df[df.scene == scene]
        X = tr[feats].fillna(0).values
        b, w = fit_weights(X, tr.y.values.astype(float), specs, model.bias, l2=5.0)
        m2 = HazardModel("crowd", b, [type(s)(**{**s.__dict__, "weight": float(wi)}) for s, wi in zip(specs, w)], model.threshold)
        p, _ = run_model(te, m2, cfg["hysteresis"])
        all_p[te.index.values] = p
        loso[f"scene{scene}"] = {"auc": auc(p, te.y.values), "weights": dict(zip(feats, map(float, w))), "bias": b}
    res["loso_refit"] = {"per_scene": loso, "pooled_auc": auc(all_p, df.y.values)}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "loso_refit"}, indent=1, default=str)[:3000])
    print("LOSO pooled AUC", res["loso_refit"]["pooled_auc"], {k: round(v["auc"], 3) for k, v in loso.items()})


if __name__ == "__main__":
    main()
