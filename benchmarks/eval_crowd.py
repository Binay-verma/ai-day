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
from edgecopilot.xai.calibration import conformal_threshold  # noqa: E402
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


def run_model(df, model, hyst_cfg, return_ema=False):
    probs, active, ema = [], [], []
    for clip, g in df.groupby("clip", sort=False):
        h = Hysteresis(**hyst_cfg)
        for _, r in g.iterrows():
            vals = {k: r[k] for k in model.specs if k in r and r[k] == r[k]}
            a = h.update("crowd", model.assess("zone", r.t, vals))
            probs.append(a.prob)
            active.append(a.active)
            ema.append(h.state["crowd"]["ema"])
    if return_ema:
        return np.array(probs), np.array(active), np.array(ema)
    return np.array(probs), np.array(active)


def conformal_events(df, ema, alpha, on_frames):
    """Per-camera calibration: for each test clip, the alert line is the conformal
    quantile of smoothed scores on the NORMAL frames of the other clips of the same
    scene (same camera), i.e. an installation-time calibration on normal footage."""
    active = np.zeros(len(df), bool)
    taus = {}
    for clip, g in df.groupby("clip"):
        cal = df[(df.scene == g.scene.iloc[0]) & (df["clip"] != clip) & (df.y == 0)].index.values
        tau = conformal_threshold(ema[cal], alpha)
        taus[int(clip)] = tau
        run = 0
        for i in g.index.values:
            run = run + 1 if ema[i] > tau else 0
            active[i] = run >= on_frames
    return active, taus


def events(df, active, fps=30.0, early_s=1.5):
    """Event metrics. The UMN banner appears up to ~1 s after people start running,
    so an alert within ``early_s`` before the labelled onset counts as an early
    detection (negative delay), not a false alarm."""
    out = []
    for clip, g in df.groupby("clip", sort=False):
        idx = g.index.values
        y = g.y.values
        act = active[idx]
        onset = np.flatnonzero(y)[0] if y.any() else None
        start = max(0, onset - int(early_s * fps)) if onset is not None else None
        pre = act[:start] if onset is not None else act
        # count rising edges of alerts during the normal part = false alarms
        fa = int(np.sum(np.diff(np.concatenate([[0], pre.astype(int)])) == 1))
        det = None
        if onset is not None:
            hits = np.flatnonzero(act[start:])
            det = (hits[0] + start - onset) / fps if len(hits) else None
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
           "feature_auc_standard_labels": {k: auc(df[k].fillna(0).values, df.y.values) for k in feats + ["speed", "fast_speed", "energy", "motion_area", "divergence"] if k in df}}
    # UMN labels come from a burnt-in banner that (a) appears ~0.5-1 s after the
    # panic starts and (b) stays on while the scene is EMPTY after everyone ran
    # off (51 % of "abnormal" frames have nobody in view).  We report the
    # standard AUC and a tolerant AUC that drops the 1 s before onset and the
    # empty-and-still abnormal frames.  Event-level numbers use standard labels.
    keep = np.ones(len(df), bool)
    for clip, g in df.groupby("clip"):
        on = g.index[g.y == 1]
        if len(on):
            keep[(df.index >= on[0] - 30) & (df.index < on[0])] = False
    empty = ((df.y == 1) & (df.people == 0) & (df.motion_area.fillna(0) < 0.01)).values
    tol = keep & ~empty
    res["tolerant_mask_frames"] = int(tol.sum())
    res["feature_auc_tolerant"] = {k: auc(df[k].fillna(0).values[tol], df.y.values[tol]) for k in res["feature_auc_standard_labels"]}
    probs, active, ema = run_model(df, model, cfg["hysteresis"], return_ema=True)
    for alpha in (0.01, 0.002):
        act_c, taus = conformal_events(df, ema, alpha, cfg["hysteresis"]["on_frames"])
        ev = events(df, act_c)
        res[f"conformal_alpha{alpha}"] = {"events": ev, "tau_logit": taus,
                                         "detected": sum(e["detected"] for e in ev),
                                         "false_alarms": sum(e["false_alarms_before_onset"] for e in ev),
                                         "normal_frames_flagged": float(act_c[(df.y == 0).values].mean())}
        print(f"conformal alpha={alpha}: detected {res[f'conformal_alpha{alpha}']['detected']}/11, false alarms {res[f'conformal_alpha{alpha}']['false_alarms']}, "
              f"delays {[e['delay_s'] for e in ev]}")
    res["expert_model"] = {"frame_auc": auc(probs, df.y.values), "frame_auc_tolerant": auc(probs[tol], df.y.values[tol]),
                           "events": events(df, active)}
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
    res["loso_refit"] = {"per_scene": loso, "pooled_auc": auc(all_p, df.y.values),
                         "pooled_auc_tolerant": auc(all_p[tol], df.y.values[tol])}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "loso_refit"}, indent=1, default=str)[:3000])
    print("LOSO pooled AUC", res["loso_refit"]["pooled_auc"], "tolerant", res["loso_refit"]["pooled_auc_tolerant"],
          {k: round(v["auc"], 3) for k, v in loso.items()})
    ev = res["expert_model"]["events"]
    print("events: detected", sum(e["detected"] for e in ev), "/", len(ev), "false alarms", sum(e["false_alarms_before_onset"] for e in ev),
          "median delay", np.median([e["delay_s"] for e in ev if e["delay_s"] is not None]))


if __name__ == "__main__":
    main()
