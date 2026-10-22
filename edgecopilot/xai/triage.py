"""Conflict-aware triage: ALARM vs VERIFY.

Not every above-threshold score deserves the same response.  Using the
evidence ledger we route an alert to

* ``ALARM``  - strong margin, little counter-evidence, and (where several
               modalities exist) camera and sensors agree;
* ``VERIFY`` - the score is over the line but the evidence is thin or
               contradictory (e.g. the camera sees "smoke" while the MQ-2
               sensor next to it reads clean air).  The operator is asked to
               check before escalating, which is where human judgement adds
               the most value.

The rule is deliberately simple and auditable; thresholds live in config.
"""
from __future__ import annotations

from typing import Dict, List

from .fusion import Assessment


def triage(a: Assessment, min_margin: float = 1.0, max_conflict: float = 0.35,
           strong: float = 0.5) -> Dict:
    pos = [c for c in a.cues if c.contribution > 0]
    neg = [c for c in a.cues if c.contribution < 0]
    s_pos = sum(c.contribution for c in pos)
    s_neg = -sum(c.contribution for c in neg)
    conflict = s_neg / s_pos if s_pos > 0 else 1.0
    margin = a.logit - a.logit_threshold
    reasons: List[str] = []
    # cross-modal disagreement: a sensor strongly against while vision is for (or vice versa)
    by_src_pos = {c.source for c in pos if c.contribution >= strong}
    disagree = [c for c in neg if -c.contribution >= strong and c.source not in by_src_pos
                and c.source in ("sensor", "vision") and by_src_pos & {"sensor", "vision"}]
    if margin < min_margin:
        reasons.append("score only just above the alert line")
    if conflict > max_conflict:
        reasons.append("substantial counter-evidence")
    for c in disagree:
        reasons.append(f"{c.source} disagrees: {c.phrase}")
    mode = "VERIFY" if reasons else "ALARM"
    return {"mode": mode, "margin_logit": margin, "conflict_ratio": conflict,
            "reasons": reasons, "conflicting_cues": [c.name for c in disagree]}
