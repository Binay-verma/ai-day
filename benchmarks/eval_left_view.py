"""'Went down and left the view' on GMDCSA24, through the real camera pipeline.

Cached pose detections are replayed through ``Copilot.process`` (a stand-in detector returns the cached
detections for each frame; fire and crowd branches off), once with the left-view rule off and once on.
Video-level result: a fall clip counts as detected if any fall alert was raised; an ADL clip with any
fall alert is a false alarm. The rule's thresholds were chosen on these clips, so this is a development
result; the held-out false-alarm check is the external and UR-Fall runs (eval_external.py, eval_urfall.py).

    python benchmarks/eval_left_view.py [--cache data/gmdcsa24_s_lowconf.pkl]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgecopilot.perception.detector import Detection, DetectorOutput  # noqa: E402
from edgecopilot.pipeline import Copilot, load_config  # noqa: E402


class Replay:
    """Stand-in pose model: returns the cached detections of the current frame."""

    def __init__(self, min_score: float):
        self.dets, self.min_score = [], min_score

    def __call__(self, frame):
        return DetectorOutput([Detection(np.array(b, float), s, 0, "person", np.array(k, float))
                               for b, s, k in self.dets if s >= self.min_score], None, (1, 0, 0, 0, 0), 0.0)


def run(items, cfg, left_view: bool, protocol: str = "clip"):
    c = copy.deepcopy(cfg)
    c["runtime"]["left_view"] = {**c["runtime"].get("left_view", {}), "enabled": left_view}
    c["runtime"]["motion_gate"] = False
    rows = []
    for it in items:
        rp = Replay(cfg["models"]["pose"]["conf"])
        cop = Copilot(c, zone="Corridor B", pose_model=rp, fire_model=None,
                      enable={"fall": True, "crowd": False, "fire_smoke": False})
        frame = np.zeros((*it["shape"], 3), np.uint8)
        alerts = []
        for fr in it["frames"]:
            rp.dets = fr["dets"]
            alerts += cop.process(frame, fr["t"]).new_alerts
        # protocol: "clip" = the clip as recorded (same as eval_fall.py); "hold" = the last frame's detections
        # are repeated for 4 s ("the person stays down", as in eval_urfall.py / eval_external.py)
        if protocol == "hold" and it["frames"]:
            t = it["frames"][-1]["t"]
            for j in range(1, 61):
                alerts += cop.process(frame, t + j / 15).new_alerts
        lv = [a for a in alerts if a.explanation.get("headline", "").startswith("Possible fall out of view")]
        rows.append({"video": f"S{it['subject']}/{'Fall' if it['label'] else 'ADL'}/{os.path.basename(it['path'])}",
                     "label": it["label"], "pred": bool(alerts), "left_view_alerts": len(lv), "desc": it["desc"]})
    return rows


def summary(rows):
    tp = sum(r["label"] and r["pred"] for r in rows)
    fp = sum((not r["label"]) and r["pred"] for r in rows)
    fn = sum(r["label"] and not r["pred"] for r in rows)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(p + r, 1e-9), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/gmdcsa24_lowconf.pkl")
    ap.add_argument("--out", default="benchmarks/results/left_view_gmdcsa24.json")
    ap.add_argument("--protocol", choices=["clip", "hold"], default="clip")
    a = ap.parse_args()
    items = pickle.load(open(a.cache, "rb"))
    cfg = load_config("configs/campus.yaml")
    off, on = run(items, cfg, False, a.protocol), run(items, cfg, True, a.protocol)
    changed = [{"video": y["video"], "label": y["label"], "desc": y["desc"]} for x, y in zip(off, on) if x["pred"] != y["pred"]]
    res = {"cache": a.cache, "protocol": a.protocol, "rule_off": summary(off), "rule_on": summary(on), "changed_videos": changed,
           "left_view_alerts_total": sum(r["left_view_alerts"] for r in on)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
