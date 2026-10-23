"""Linear vs monotone-GAM fall model, leave-one-subject-out on GMDCSA24.

Both models are exactly additive (explanations stay exact). The GAM replaces each
cue's straight line with a learned monotone piecewise-linear shape (4 segments at
training-data quantiles, slopes pulled towards the expert line). Hyper-parameters
(n_seg = 4, l2 = 3) are fixed in advance and not tuned on the test subjects; a
sensitivity sweep is reported separately so the choice can be inspected.

    python benchmarks/eval_fall_gam.py --cache data/gmdcsa24_s_lowconf.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_fall import cue_series, evaluate, frame_labels, metrics, refit  # noqa: E402

from edgecopilot.pipeline import load_config  # noqa: E402
from edgecopilot.xai.calibration import conformal_threshold  # noqa: E402
from edgecopilot.xai.fusion import HazardModel, fit_gam  # noqa: E402


def design(model, items, series_all):
    names = list(model.specs)
    X, y = [], []
    for it, s in zip(items, series_all):
        for (t, tid, vals, rel), lab in zip(s, frame_labels(it, s)):
            X.append([vals.get(n, model.specs[n].ref) for n in names])
            y.append(lab)
    return np.array(X, float), np.array(y, float)


def gam_model(expert, items, series_all, n_seg, l2):
    X, y = design(expert, items, series_all)
    b, specs = fit_gam(X, y, list(expert.specs.values()), expert.bias, n_seg=n_seg, l2=l2)
    return HazardModel("fall", b, specs, expert.threshold)


def loso(items, series_all, expert, hyst, fit, alpha=0.1):
    rows, crow = [], []
    for s in (1, 2, 3, 4):
        tr = [i for i, it in enumerate(items) if it["subject"] != s]
        te = [i for i, it in enumerate(items) if it["subject"] == s]
        m = fit([items[i] for i in tr], [series_all[i] for i in tr])
        rows += evaluate([items[i] for i in te], [series_all[i] for i in te], m, hyst)
        r_tr = evaluate([items[i] for i in tr], [series_all[i] for i in tr], m, hyst)
        tau = conformal_threshold(np.array([r["peak_logit"] for r in r_tr if not r["label"]]), alpha)
        crow += evaluate([items[i] for i in te], [series_all[i] for i in te], m, hyst, threshold_logit=tau)
    adl = [r for r in crow if not r["label"]]
    return metrics(rows), metrics(crow), sum(r["pred"] for r in adl) / max(len(adl), 1), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/gmdcsa24_s_lowconf.pkl")
    ap.add_argument("--config", default="configs/campus.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    items = pickle.load(open(a.cache, "rb"))
    cfg = load_config(a.config)
    hyst = cfg["hysteresis"]
    expert = HazardModel.from_config("fall", cfg["hazards"]["fall"])
    series_all = [cue_series(it, min_score=cfg["models"]["pose"]["conf"]) for it in items]
    res = {"cache": a.cache}
    lin = loso(items, series_all, expert, hyst, lambda i, s: refit(expert, i, s))
    gam = loso(items, series_all, expert, hyst, lambda i, s: gam_model(expert, i, s, 4, 3.0))
    for name, r in (("linear_monotone", lin), ("gam_monotone", gam)):
        res[name] = {"loso": r[0], "loso_conformal": r[1], "conformal_far": r[2]}
        print(f"{name:16s} F1 {r[0]['f1']:.3f} P {r[0]['precision']:.3f} R {r[0]['recall']:.3f} | "
              f"conformal F1 {r[1]['f1']:.3f} FAR {r[2]:.3f}")
    # paired comparison on the same videos
    lr = {x["video"]: x["pred"] == bool(x["label"]) for x in lin[3]}
    gr = {x["video"]: x["pred"] == bool(x["label"]) for x in gam[3]}
    res["discordant"] = {"gam_right_linear_wrong": sum(gr[v] and not lr[v] for v in lr),
                         "linear_right_gam_wrong": sum(lr[v] and not gr[v] for v in lr)}
    print("discordant", res["discordant"])
    if a.sweep:
        res["sweep"] = {}
        for n_seg in (3, 4, 6):
            for l2 in (1.0, 3.0, 10.0):
                r = loso(items, series_all, expert, hyst, lambda i, s, n=n_seg, reg=l2: gam_model(expert, i, s, n, reg))
                res["sweep"][f"seg{n_seg}_l2{l2}"] = r[0]["f1"]
                print(f"  sweep seg={n_seg} l2={l2}: F1 {r[0]['f1']:.3f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
