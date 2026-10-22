"""Closed-form counterfactuals and sufficient/necessary reasons for alerts.

Because :class:`~edgecopilot.xai.fusion.HazardModel` is additive in log-odds,
three classic explanation questions have exact answers (no search, no
sampling, < 0.1 ms):

* **Counterfactual** - "the alert would NOT have fired if smoke density had
  stayed below 0.21": for each positive cue, the value it would need for the
  logit to drop under the threshold with everything else unchanged.
* **Sufficient reason** - the smallest set of cues that on their own (all
  other cues at their reference value) already exceed the threshold.
  Greedy-by-contribution is provably minimal in cardinality here.
* **Necessary cues** - cues whose removal alone flips the decision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from .fusion import Assessment, HazardModel


@dataclass
class Counterfactual:
    cue: str
    current: float
    needed: float
    unit: str
    direction: str  # "below" | "above"

    def text(self) -> str:
        u = f" {self.unit}" if self.unit else ""
        return (f"{self.cue.replace('_', ' ')} had been {self.direction} "
                f"{self.needed:.2f}{u} (observed {self.current:.2f}{u})")


def counterfactuals(a: Assessment, model: HazardModel, margin: float = 1e-3) -> List[Counterfactual]:
    """Single-cue counterfactuals that would flip an active (or above-threshold) alert."""
    tau = a.logit_threshold
    excess = a.logit - tau
    out: List[Counterfactual] = []
    if excess <= 0:
        return out
    for cue in a.cues:
        if cue.contribution <= 0:
            continue
        spec = model.specs.get(cue.name)
        if spec is None or spec.weight == 0 or cue.reliability == 0:
            continue
        target = cue.contribution - excess - margin  # contribution needed
        x_needed = spec.inverse(target / (cue.reliability * spec.weight))
        lo, hi = min(spec.lo, spec.hi), max(spec.lo, spec.hi)
        if not (lo <= x_needed <= hi):
            continue  # this cue alone cannot flip the decision
        direction = "below" if spec.weight > 0 else "above"
        out.append(Counterfactual(cue.name, cue.value, x_needed, spec.unit, direction))
    # most plausible first: the smallest change relative to the cue's range
    def rel_change(cf: Counterfactual) -> float:
        sp = model.specs[cf.cue]
        return abs(cf.needed - cf.current) / (abs(sp.hi - sp.lo) + 1e-9)
    return sorted(out, key=rel_change)


def sufficient_reason(a: Assessment) -> Optional[List[str]]:
    """Smallest cue subset whose contributions alone reach the threshold."""
    tau = a.logit_threshold
    pos = sorted([c for c in a.cues if c.contribution > 0], key=lambda c: -c.contribution)
    acc = a.bias
    chosen: List[str] = []
    for c in pos:
        if acc >= tau:
            break
        acc += c.contribution
        chosen.append(c.name)
    return chosen if acc >= tau else None


def necessary_cues(a: Assessment) -> List[str]:
    tau = a.logit_threshold
    if a.logit < tau:
        return []
    return [c.name for c in a.cues if c.contribution > 0 and a.logit - c.contribution < tau]


def explain_decision(a: Assessment, model: HazardModel) -> Dict:
    return {
        "counterfactuals": [cf.__dict__ | {"text": cf.text()} for cf in counterfactuals(a, model)],
        "sufficient_reason": sufficient_reason(a),
        "necessary_cues": necessary_cues(a),
        "margin_logit": a.logit - a.logit_threshold,
        "confidence": a.prob,
        "odds_multiplier": {c.name: math.exp(c.contribution) for c in a.cues},
    }
