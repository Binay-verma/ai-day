"""Probability calibration of the fall model (does "80 % confidence" mean ~80 %?).

Leave-one-subject-out on GMDCSA24: for each held-out subject, the additive model is
re-fitted on the other three (as in eval_fall.py) and its *video-level* confidence (peak
smoothed probability over the clip) is compared with the truth (fall clip or not).
Reported: expected calibration error (ECE, 15 equal-width bins, Guo et al. 2017), Brier
score, and the same after temperature scaling fitted on the training subjects only.
A reliability table is saved for plotting.

    python benchmarks/eval_calibration.py --cache data/gmdcsa24_s_lowconf.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_fall import cue_series, evaluate, refit  # noqa: E402

from edgecopilot.pipeline import load_config  # noqa: E402
from edgecopilot.xai.fusion import HazardModel  # noqa: E402


def ece(p, y, bins=15):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    e, table = 0.0, []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
            table.append({"bin": [round(edges[i], 3), round(edges[i + 1], 3)], "n": int(m.sum()),
                          "confidence": float(p[m].mean()), "accuracy": float(y[m].mean())})
    return float(e), table


def sig(z):
    return 1 / (1 + np.exp(-z))


def fit_temperature(z, y):
    z, y = np.asarray(z, float), np.asarray(y, float)

    def nll(T):
        p = np.clip(sig(z / T), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/gmdcsa24_s_lowconf.pkl")
    ap.add_argument("--out", default="benchmarks/results/calibration_fall.json")
    a = ap.parse_args()
    items = pickle.load(open(a.cache, "rb"))
    cfg = load_config("configs/campus.yaml")
    hyst = cfg["hysteresis"]
    expert = HazardModel.from_config("fall", cfg["hazards"]["fall"])
    series_all = [cue_series(it, min_score=cfg["models"]["pose"]["conf"]) for it in items]
    z_te, y_te, z_ts = [], [], []
    for s in (1, 2, 3, 4):
        tr = [i for i, it in enumerate(items) if it["subject"] != s]
        te = [i for i, it in enumerate(items) if it["subject"] == s]
        m = refit(expert, [items[i] for i in tr], [series_all[i] for i in tr])
        r_tr = evaluate([items[i] for i in tr], [series_all[i] for i in tr], m, hyst)
        r_te = evaluate([items[i] for i in te], [series_all[i] for i in te], m, hyst)
        ztr = np.array([max(r["peak_logit"], -30) for r in r_tr])
        T = fit_temperature(ztr, [r["label"] for r in r_tr])
        for r in r_te:
            z = max(r["peak_logit"], -30)
            z_te.append(z)
            z_ts.append(z / T)
            y_te.append(r["label"])
    p_raw, p_ts = sig(np.array(z_te)), sig(np.array(z_ts))
    y = np.array(y_te, float)
    e_raw, t_raw = ece(p_raw, y)
    e_ts, t_ts = ece(p_ts, y)
    res = {"cache": a.cache, "n_videos": len(y), "level": "video (peak smoothed probability)",
           "ece_raw": e_raw, "brier_raw": float(np.mean((p_raw - y) ** 2)),
           "ece_temperature_scaled": e_ts, "brier_temperature_scaled": float(np.mean((p_ts - y) ** 2)),
           "reliability_raw": t_raw, "reliability_temperature_scaled": t_ts}
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("reliability")}, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
