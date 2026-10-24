"""Corroboration-constrained re-fit (min_cues = 2) on GMDCSA24, LOSO, YOLO11s-pose features.

Development-set check only: does forbidding single-cue alarms during fitting keep most of the
re-fit's F1 gain? (The unconstrained re-fit put nearly all weight on descent speed and did not
transfer to other footage - RESULTS section 8.)
"""
import math
import pickle
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "benchmarks")
from eval_fall import cue_series, evaluate, metrics, refit  # noqa: E402

from edgecopilot.pipeline import load_config  # noqa: E402
from edgecopilot.xai.fusion import HazardModel  # noqa: E402

items = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/gmdcsa24_s_lowconf.pkl", "rb"))
cfg = load_config("configs/campus.yaml")
hyst = cfg["hysteresis"]
expert = HazardModel.from_config("fall", cfg["hazards"]["fall"])
line = math.log(0.5 / 0.5)  # fall min_threshold 0.5
S = [cue_series(it, min_score=cfg["models"]["pose"]["conf"]) for it in items]
for name, scl in (("unconstrained", None), ("min_cues_2", line)):
    rows, ws = [], []
    for s in (1, 2, 3, 4):
        tr = [i for i, it in enumerate(items) if it["subject"] != s]
        te = [i for i, it in enumerate(items) if it["subject"] == s]
        m = refit(expert, [items[i] for i in tr], [S[i] for i in tr], single_cue_line=scl)
        rows += evaluate([items[i] for i in te], [S[i] for i in te], m, hyst)
        ws.append({n: round(sp.weight, 3) for n, sp in m.specs.items()})
    mm = metrics(rows)
    print(name, "F1", round(mm["f1"], 3), "P", round(mm["precision"], 3), "R", round(mm["recall"], 3), "fold-1 weights", ws[0], flush=True)
