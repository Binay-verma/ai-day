"""GMDCSA24 fall benchmark (video-level, leave-one-subject-out).

Uses the pose detections cached by ``extract_fall_features.py`` and replays
the *exact* production chain (tracker -> fall cues -> additive model ->
hysteresis) on them.  Three variants are reported:

* ``expert``   - weights straight from configs/campus.yaml (no training data)
* ``loso_fit`` - weights re-fitted on 3 subjects, tested on the 4th (x4 folds)
* ``conformal``- expert/fitted model whose alert threshold is the split-
                 conformal quantile of per-video peak scores on the training
                 subjects' ADL clips, i.e. a stated false-alarm budget alpha.

Metrics: accuracy, precision, recall, F1, specificity, detection delay after
the annotated fall onset, and empirical false-alarm rate vs alpha.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgecopilot.perception.detector import Detection  # noqa: E402
from edgecopilot.perception.fall import FallFeatureExtractor  # noqa: E402
from edgecopilot.perception.tracker import Tracker  # noqa: E402
from edgecopilot.pipeline import load_config  # noqa: E402
from edgecopilot.xai.calibration import conformal_threshold  # noqa: E402
from edgecopilot.xai.fusion import CueSpec, HazardModel, Hysteresis, fit_weights, sigmoid  # noqa: E402


def cue_series(item, fx_kwargs=None, min_score=0.35):
    """Replay tracker + fall cue extraction -> list of (t, track, values, rel)."""
    tracker, fx = Tracker(), FallFeatureExtractor(**(fx_kwargs or {}))
    out = []
    for fr in item["frames"]:
        dets = [Detection(np.array(b, float), s, 0, "person", np.array(k, float)) for b, s, k in fr["dets"] if s >= min_score]
        for tr in tracker.update(dets, fr["t"]):
            if tr.hits < 2:
                continue
            vals, rel = fx(tr, fr["t"])
            out.append((fr["t"], tr.track_id, vals, rel))
    return out


def run_video(series, model, hyst_cfg):
    """Return (alert_time or None, peak_logit, per-row logits)."""
    h = Hysteresis(**hyst_cfg)
    first = None
    peak = -math.inf
    logits = []
    for t, tid, vals, rel in series:
        a = model.assess(f"person#{tid}", t, vals, rel)
        h.update(f"fall:{tid}", a)
        st = h.state[f"fall:{tid}"]
        peak = max(peak, st["ema"])
        logits.append(a.logit)
        if a.onset and first is None:
            first = t
    return first, peak, logits


def metrics(rows):
    tp = sum(1 for r in rows if r["label"] and r["pred"])
    fp = sum(1 for r in rows if not r["label"] and r["pred"])
    fn = sum(1 for r in rows if r["label"] and not r["pred"])
    tn = sum(1 for r in rows if not r["label"] and not r["pred"])
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    delays = [r["delay"] for r in rows if r["label"] and r["pred"] and r["delay"] is not None]
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / max(len(rows), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9), "specificity": tn / max(tn + fp, 1),
            "delay_median_s": float(np.median(delays)) if delays else None,
            "delay_p90_s": float(np.percentile(delays, 90)) if delays else None}


def frame_labels(item, series):
    """Frame-level labels for fitting: 1 inside annotated fall interval (+1 s), 0 for ADL clips."""
    if not item["label"]:
        return [0] * len(series)
    iv = [(a, b) for n, a, b in item["intervals"] if n.lower().startswith("fall")]
    return [int(any(a <= t <= b + 1.0 for a, b in iv)) for t, *_ in series]


def evaluate(items, series_all, model, hyst, threshold_logit=None):
    rows = []
    for it, s in zip(items, series_all):
        m = deepcopy(model)
        if threshold_logit is not None:
            m.threshold = sigmoid(threshold_logit) if threshold_logit < 50 else 1.0
        first, peak, _ = run_video(s, m, hyst)
        delay = None
        if first is not None and it["fall_onset"] is not None:
            delay = first - it["fall_onset"]
        rows.append({"video": f"S{it['subject']}/{'Fall' if it['label'] else 'ADL'}/{os.path.basename(it['path'])}",
                     "subject": it["subject"], "label": it["label"], "pred": first is not None,
                     "alert_t": first, "delay": delay, "peak_logit": peak, "desc": it["desc"]})
    return rows


def refit(model, items, series_all, l2=3.0):
    names = list(model.specs)
    X, y = [], []
    for it, s in zip(items, series_all):
        for (t, tid, vals, rel), lab in zip(s, frame_labels(it, s)):
            X.append([vals.get(n, model.specs[n].ref) for n in names])
            y.append(lab)
    specs = [model.specs[n] for n in names]
    b, w = fit_weights(np.array(X), np.array(y, float), specs, model.bias, l2=l2)
    m2 = HazardModel("fall", b, [CueSpec(**{**s.__dict__, "weight": float(wi)}) for s, wi in zip(specs, w)], model.threshold)
    return m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/gmdcsa24_cues.pkl")
    ap.add_argument("--config", default="configs/campus.yaml")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--min-score", type=float, default=None, help="person confidence (default: config pose conf)")
    ap.add_argument("--out", default="benchmarks/results/fall_gmdcsa24.json")
    args = ap.parse_args()
    items = pickle.load(open(args.cache, "rb"))
    cfg = load_config(args.config)
    hyst = cfg["hysteresis"]
    expert = HazardModel.from_config("fall", cfg["hazards"]["fall"])
    min_score = args.min_score if args.min_score is not None else cfg["models"]["pose"]["conf"]
    series_all = [cue_series(it, min_score=min_score) for it in items]
    res = {"dataset": "GMDCSA24", "videos": len(items), "falls": sum(i["label"] for i in items), "min_score": min_score}

    rows = evaluate(items, series_all, expert, hyst)
    res["expert"] = {"overall": metrics(rows), "per_subject": {s: metrics([r for r in rows if r["subject"] == s]) for s in (1, 2, 3, 4)}}
    res["expert"]["errors"] = [{k: r[k] for k in ("video", "label", "pred", "delay", "desc")} for r in rows if r["label"] != r["pred"]]

    loso_rows, conf_rows, folds = [], [], {}
    for s in (1, 2, 3, 4):
        tr_i = [i for i, it in enumerate(items) if it["subject"] != s]
        te_i = [i for i, it in enumerate(items) if it["subject"] == s]
        m2 = refit(expert, [items[i] for i in tr_i], [series_all[i] for i in tr_i])
        r_te = evaluate([items[i] for i in te_i], [series_all[i] for i in te_i], m2, hyst)
        loso_rows += r_te
        # conformal budget: peak EMA-logit on training ADL clips
        r_tr = evaluate([items[i] for i in tr_i], [series_all[i] for i in tr_i], m2, hyst)
        normal = [r["peak_logit"] for r in r_tr if not r["label"]]
        tau = conformal_threshold(np.array(normal), args.alpha)
        r_c = evaluate([items[i] for i in te_i], [series_all[i] for i in te_i], m2, hyst, threshold_logit=tau)
        conf_rows += r_c
        folds[f"test_subject_{s}"] = {"bias": m2.bias, "weights": {n: sp.weight for n, sp in m2.specs.items()},
                                      "loso": metrics(r_te), "conformal_tau_logit": tau, "conformal": metrics(r_c)}
    res["loso_fit"] = {"overall": metrics(loso_rows), "folds": folds,
                       "errors": [{k: r[k] for k in ("video", "label", "pred", "delay", "desc")} for r in loso_rows if r["label"] != r["pred"]]}
    cm = metrics(conf_rows)
    adl = [r for r in conf_rows if not r["label"]]
    res["conformal"] = {"alpha": args.alpha, "overall": cm,
                        "empirical_far": sum(r["pred"] for r in adl) / max(len(adl), 1)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, default=float)
    show = lambda m: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}
    print("expert  ", show(res["expert"]["overall"]))
    print("loso_fit", show(res["loso_fit"]["overall"]))
    print("conformal", show(cm), "FAR", res["conformal"]["empirical_far"], "alpha", args.alpha)
    print("expert errors:")
    for e in res["expert"]["errors"]:
        print("  ", e)


if __name__ == "__main__":
    main()
