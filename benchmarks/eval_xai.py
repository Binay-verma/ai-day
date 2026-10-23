"""Explanation-quality benchmark on real alerts from the fall and crowd benchmarks.

For every alert raised on GMDCSA24 (falls) and UMN (crowd) we measure:

1. Attribution exactness   |bias + sum(contributions) - logit|  (should be ~1e-15)
2. Deletion test           setting a cited cue to its reference changes the logit
                           by exactly its contribution
3. Counterfactuals         validity (re-scored decision flips), sparsity (1 cue),
                           plausibility (needed value within the 5-95 % range that
                           cue takes on normal data)
4. Grounding of templates  unsupported-claim rate of the generated explanation
5. Verifier stress test    a "noisy narrator" rewrites each explanation with the
                           error types LLM narrators make (number drift, wrong
                           cue cited as cause, wrong hazard, invented sensor
                           reading, unit/percent confusion) and with meaning-
                           preserving paraphrases; we report detection rate and
                           false-rejection rate of the verifier.
6. Latency                 explanation + counterfactual + verification time
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.eval_fall import cue_series  # noqa: E402
from edgecopilot.copilot.recommender import Recommender  # noqa: E402
from edgecopilot.pipeline import load_config  # noqa: E402
from edgecopilot.xai.counterfactual import counterfactuals, explain_decision, sufficient_reason  # noqa: E402
from edgecopilot.xai.fusion import HazardModel, Hysteresis  # noqa: E402
from edgecopilot.xai.nlg import NUM_RE, explain, numbers_in, verify  # noqa: E402

PARAPHRASES = [
    (r"Alert raised due to", "The alert was triggered because of"),
    (r"Alert raised due to", "Raised because of"),
    (r"It would not have fired if", "Had it not been the case that"),
    (r"Confidence (\d+)%", r"Estimated probability \1%"),
]


def collect_alerts(cfg, fall_cache, umn_cache):
    """Replay the benchmarks and return assessments at alert onset (+ all normal-data cue values)."""
    out, normal_vals = [], {"fall": [], "crowd": []}
    if os.path.exists(fall_cache):
        m = HazardModel.from_config("fall", cfg["hazards"]["fall"])
        for it in pickle.load(open(fall_cache, "rb")):
            h = Hysteresis(**cfg["hysteresis"])
            for t, tid, vals, rel in cue_series(it, min_score=cfg["models"]["pose"]["conf"]):
                a = h.update(f"f{tid}", m.assess(f"person#{tid}", t, vals, rel))
                if not it["label"]:
                    normal_vals["fall"].append(vals)
                if a.onset:
                    out.append((m, a, vals, rel, "Corridor B"))
    if os.path.exists(umn_cache):
        m = HazardModel.from_config("crowd", cfg["hazards"]["crowd"])
        df = pd.read_pickle(umn_cache)
        for clip, g in df.groupby("clip"):
            h = Hysteresis(**cfg["hysteresis"])
            for _, r in g.iterrows():
                vals = {k: r[k] for k in m.specs if k in r and r[k] == r[k]}
                a = h.update("c", m.assess("Auditorium", r.t, vals))
                if r.y == 0:
                    normal_vals["crowd"].append(vals)
                if a.onset:
                    out.append((m, a, vals, {}, "Auditorium"))
    return out, normal_vals


def perturb(text, a, model, rng):
    """Return list of (kind, corrupted_text) - each must be rejected."""
    outs = []
    nums = [m for m in NUM_RE.finditer(text)]
    cue_nums = [m for m in nums if not text[max(0, m.start() - 12):m.start()].lower().strip().endswith(("confidence", "threshold"))]
    if cue_nums:
        m = rng.choice(cue_nums)
        raw = m.group(0)
        pct = raw.endswith("%")
        x = float(raw.rstrip("%"))
        new = x * rng.choice([0.5, 0.6, 1.5, 1.8]) + (7 if abs(x) < 1 else 0)
        rep = (f"{new:.0f}%" if pct else (f"{new:.1f}" if "." in raw else f"{new:.0f}"))
        if rep != raw:
            outs.append(("number_drift", text[:m.start()] + rep + text[m.end():]))
    neg = [c for c in a.cues if c.contribution < 0]
    if neg and "due to" in text:
        c = rng.choice(neg)
        outs.append(("false_cause", text.replace("Alert raised due to ", f"Alert raised due to {c.phrase}, ", 1)))
    other = {"fall": "Smoke detected", "crowd": "Fall detected", "fire_smoke": "Fall detected"}[a.hazard]
    first = text.split(".")[0]
    outs.append(("wrong_hazard", text.replace(first, other + (" " + first.split(" ", 2)[-1] if " " in first else ""), 1)))
    outs.append(("invented_sensor", text.replace("Alert raised due to ", "Alert raised due to the MQ-2 sensor reading 640 ppm, ", 1)))
    pct_m = re.search(r"Confidence (\d+)%", text)
    if pct_m:
        outs.append(("confidence_inflation", text.replace(pct_m.group(0), f"Confidence {min(99, int(pct_m.group(1)) + 25) if int(pct_m.group(1)) < 74 else int(pct_m.group(1)) - 30}%")))
    return outs


def paraphrase(text, rng):
    outs = []
    for pat, rep in PARAPHRASES:
        if re.search(pat, text):
            outs.append(re.sub(pat, rep, text, count=1))
    # drop decimals the way a narrator rounds ("1.8" -> "about 2")
    m = re.search(r"(\d+)\.(\d)\b", text)
    if m:
        v = float(m.group(0))
        outs.append(text[:m.start()] + f"{round(v):d}" + text[m.end():])
    # reorder: move the confidence sentence first
    sents = re.split(r"(?<=\.)\s+", text)
    if len(sents) > 2:
        outs.append(" ".join([sents[-1]] + sents[:-1]))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/campus.yaml")
    ap.add_argument("--fall-cache", default="data/gmdcsa24_lowconf.pkl")
    ap.add_argument("--umn-cache", default="data/umn_features.pkl")
    ap.add_argument("--out", default="benchmarks/results/xai.json")
    args = ap.parse_args()
    rng = random.Random(0)
    cfg = load_config(args.config)
    rec = Recommender.from_config(cfg)
    alerts, normal_vals = collect_alerts(cfg, args.fall_cache, args.umn_cache)
    print(f"{len(alerts)} alerts collected")
    res = {"n_alerts": len(alerts)}
    attr_err, del_err, cf_valid, cf_total, cf_plaus, n_with_cf, suff_sizes = [], [], 0, 0, 0, 0, []
    ground, n_claims, lat = [], [], []
    stress = {}
    para_ok, para_n = 0, 0
    ranges = {}
    for hz, rows in normal_vals.items():
        if rows:
            df = pd.DataFrame(rows)
            ranges[hz] = {c: (float(df[c].quantile(0.05)), float(df[c].quantile(0.95))) for c in df.columns}
    for m, a, vals, rel, loc in alerts:
        attr_err.append(a.attribution_error())
        for c in a.cues:
            v2 = dict(vals)
            v2[c.name] = m.specs[c.name].ref
            b = m.assess(a.subject, a.t, v2, rel)
            del_err.append(abs((a.logit - b.logit) - c.contribution))
        t0 = time.perf_counter()
        recs = [x["text"] for x in rec.recommend(a, loc)["actions"][:3]]
        e = explain(a, m, loc, recs)
        v = verify(e.text, a, m, loc, extra_numbers=numbers_in(recs))
        explain_decision(a, m)
        lat.append((time.perf_counter() - t0) * 1e3)
        ground.append(v.groundedness)
        n_claims.append(v.n)
        cfs = counterfactuals(a, m)
        n_with_cf += bool(cfs)
        s = sufficient_reason(a)
        if s:
            suff_sizes.append(len(s))
        for cf in cfs:
            cf_total += 1
            v2 = dict(vals)
            v2[cf.cue] = cf.needed
            b = m.assess(a.subject, a.t, v2, rel)
            cf_valid += int(b.logit < b.logit_threshold)
            lo, hi = ranges.get(a.hazard, {}).get(cf.cue, (-np.inf, np.inf))
            cf_plaus += int(lo <= cf.needed <= hi)
        for kind, bad in perturb(e.text, a, m, rng):
            st = stress.setdefault(kind, [0, 0])
            st[1] += 1
            st[0] += int(not verify(bad, a, m, loc, extra_numbers=numbers_in(recs)).ok)
        for good in paraphrase(e.text, rng):
            para_n += 1
            para_ok += int(verify(good, a, m, loc, extra_numbers=numbers_in(recs)).ok)
    res["attribution_max_abs_error"] = float(max(attr_err)) if attr_err else None
    res["deletion_test_max_abs_error"] = float(max(del_err)) if del_err else None
    res["counterfactuals"] = {"alerts_with_single_cue_cf": n_with_cf, "total": cf_total,
                              "validity": cf_valid / max(cf_total, 1), "plausibility_in_normal_range": cf_plaus / max(cf_total, 1),
                              "sufficient_reason_mean_size": float(np.mean(suff_sizes)) if suff_sizes else None}
    res["template_grounding"] = {"mean_groundedness": float(np.mean(ground)) if ground else None,
                                 "unsupported_claim_rate": float(1 - np.mean(ground)) if ground else None,
                                 "mean_claims_per_explanation": float(np.mean(n_claims)) if n_claims else None}
    res["verifier_stress"] = {k: {"detected": d, "n": n, "rate": d / max(n, 1)} for k, (d, n) in stress.items()}
    tot_d = sum(d for d, n in stress.values())
    tot_n = sum(n for d, n in stress.values())
    res["verifier_detection_rate"] = tot_d / max(tot_n, 1)
    res["verifier_false_reject_rate_on_paraphrases"] = 1 - para_ok / max(para_n, 1)
    res["n_paraphrases"] = para_n
    res["explain_latency_ms"] = {"p50": float(np.median(lat)) if lat else None, "p95": float(np.percentile(lat, 95)) if lat else None}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
