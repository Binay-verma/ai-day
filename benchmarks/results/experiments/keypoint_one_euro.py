"""Negative result: One-Euro key-point smoothing before the fall cues (see README.md here).

The OneEuro filter was removed from the tracker afterwards; this script documents the
experiment and needs that code to rerun (git history, commit before "record negative results").
"""
import pickle
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "benchmarks")
from eval_fall import cue_series, evaluate, metrics, refit  # noqa: E402

from edgecopilot.perception.tracker import OneEuro  # type: ignore[attr-defined]  # noqa: E402
from edgecopilot.pipeline import load_config  # noqa: E402
from edgecopilot.xai.fusion import HazardModel  # noqa: E402

items = pickle.load(open(sys.argv[1], "rb"))
cfg = load_config("configs/campus.yaml")
hyst = cfg["hysteresis"]
expert = HazardModel.from_config("fall", cfg["hazards"]["fall"])
for name, tk in [("none", None), ("oe_1.0_0.05", {"kp_filter": OneEuro(1.0, 0.05)}),
                 ("oe_0.5_0.01", {"kp_filter": OneEuro(0.5, 0.01)}), ("oe_2.0_0.1", {"kp_filter": OneEuro(2.0, 0.1)})]:
    S = [cue_series(it, min_score=cfg["models"]["pose"]["conf"], tracker_kwargs=tk) for it in items]
    e = metrics(evaluate(items, S, expert, hyst))["f1"]
    lr = []
    for s in (1, 2, 3, 4):
        tr = [i for i, it in enumerate(items) if it["subject"] != s]
        te = [i for i, it in enumerate(items) if it["subject"] == s]
        m = refit(expert, [items[i] for i in tr], [S[i] for i in tr])
        lr += evaluate([items[i] for i in te], [S[i] for i in te], m, hyst)
    print(name, "expert F1", round(e, 3), "LOSO refit F1", round(metrics(lr)["f1"], 3), flush=True)
