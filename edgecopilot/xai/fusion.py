"""Additive evidence fusion with exact, faithful-by-construction attributions.

Every hazard (fall / fire_smoke / crowd) is scored by an additive log-odds
model over interpretable cues:

    logit = bias + sum_i  r_i * w_i * (clip(x_i, lo_i, hi_i) - ref_i)
    p     = sigmoid(logit)

``ref_i`` is the "nothing unusual" reference value of cue i and ``r_i`` in
[0, 1] is a reliability factor (e.g. key-point visibility, sensor health).
Because the model is additive, the per-cue term *is* the exact Shapley value
of that cue relative to the reference point (Lundberg & Lee 2017: for linear
models with independent features SHAP = w_i (x_i - E[x_i])).  So the
explanation is not a post-hoc approximation of the decision - it is the
decision, decomposed.  This also gives closed-form counterfactuals and
minimal sufficient reasons (see :mod:`edgecopilot.xai.counterfactual`).

Weights are expert-initialised from the literature and can be re-fitted on
data with :func:`fit_weights` (L2-regularised logistic regression pulled
towards the expert prior).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional

import numpy as np


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class CueSpec:
    name: str
    weight: float
    ref: float
    lo: float
    hi: float
    unit: str = ""
    phrase: str = "{name} = {v:.2f}"  # human phrase, formatted with v=value
    source: str = "vision"  # vision | sensor | context
    keywords: List[str] = field(default_factory=list)  # words the NLG verifier binds to this cue

    def __post_init__(self) -> None:
        if not self.keywords:
            self.keywords = [w for w in self.name.split("_") if len(w) > 2]

    def transform(self, x: float) -> float:
        return float(np.clip(x, self.lo, self.hi)) - self.ref

    def inverse(self, delta: float) -> float:
        """Value x such that transform(x) == delta (un-clipped)."""
        return delta + self.ref


@dataclass
class Cue:
    name: str
    value: float
    unit: str
    contribution: float
    reliability: float
    phrase: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Assessment:
    hazard: str
    subject: str  # zone name or track id ("person#3")
    t: float
    bias: float
    logit: float
    prob: float
    threshold: float  # probability threshold in force
    cues: List[Cue] = field(default_factory=list)
    active: bool = False  # after temporal hysteresis
    onset: bool = False   # became active on this update

    @property
    def logit_threshold(self) -> float:
        th = min(max(self.threshold, 1e-6), 1 - 1e-6)
        return math.log(th / (1 - th))

    def attribution_error(self) -> float:
        """|bias + sum(contributions) - logit| (must be ~0: exactness check)."""
        return abs(self.bias + sum(c.contribution for c in self.cues) - self.logit)

    def top_cues(self, k: int = 3, positive_only: bool = True) -> List[Cue]:
        cs = [c for c in self.cues if (c.contribution > 0 or not positive_only)]
        return sorted(cs, key=lambda c: -abs(c.contribution))[:k]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cues"] = [c.to_dict() for c in self.cues]
        return d


class HazardModel:
    """Additive log-odds model for one hazard type."""

    def __init__(self, hazard: str, bias: float, cues: Iterable[CueSpec], threshold: float = 0.5):
        self.hazard = hazard
        self.bias = float(bias)
        self.specs: Dict[str, CueSpec] = {c.name: c for c in cues}
        self.threshold = float(threshold)
        # per-subject bias offsets learned from operator feedback
        self.bias_offset: Dict[str, float] = {}

    @classmethod
    def from_config(cls, hazard: str, cfg: dict) -> "HazardModel":
        specs = [CueSpec(name=k, **v) for k, v in cfg["cues"].items()]
        return cls(hazard, cfg.get("bias", 0.0), specs, cfg.get("threshold", 0.5))

    def assess(self, subject: str, t: float, values: Dict[str, float],
               reliability: Optional[Dict[str, float]] = None,
               zone_key: Optional[str] = None) -> Assessment:
        reliability = reliability or {}
        bias = self.bias + self.bias_offset.get(zone_key or subject, 0.0)
        cues: List[Cue] = []
        total = bias
        for name, spec in self.specs.items():
            if name not in values or values[name] is None or not np.isfinite(values[name]):
                continue  # missing cue => contributes 0 (its reference value)
            x = float(values[name])
            r = float(np.clip(reliability.get(name, 1.0), 0.0, 1.0))
            c = r * spec.weight * spec.transform(x)
            total += c
            try:
                phrase = spec.phrase.format(v=x, name=name, unit=spec.unit)
            except (KeyError, ValueError):
                phrase = f"{name} = {x:.2f}"
            cues.append(Cue(name, x, spec.unit, c, r, phrase, spec.source))
        return Assessment(self.hazard, subject, t, bias, total, sigmoid(total), self.threshold, cues)

    # ----------------------------------------------------- operator feedback
    def feedback(self, zone_key: str, label: int, prob: float, lr: float = 0.5,
                 max_shift: float = 2.0) -> float:
        """Online logistic update of the zone's intercept from operator labels.

        label=1 (confirmed) / 0 (false alarm).  Gradient of the log-loss
        w.r.t. the bias is (p - y); the shift is bounded so a single operator
        can never silence a hazard entirely.
        """
        cur = self.bias_offset.get(zone_key, 0.0)
        new = float(np.clip(cur - lr * (prob - label), -max_shift, max_shift))
        self.bias_offset[zone_key] = new
        return new


class Hysteresis:
    """Temporal smoothing: EMA on the logit + on/off hysteresis + min duration."""

    def __init__(self, alpha: float = 0.5, on_frames: int = 3, off_frames: int = 10, off_margin: float = 0.15):
        self.alpha = alpha
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.off_margin = off_margin
        self.state: Dict[str, dict] = {}

    def reset(self) -> None:
        self.state.clear()

    def update(self, key: str, a: Assessment) -> Assessment:
        st = self.state.setdefault(key, {"ema": a.logit, "above": 0, "below": 0, "active": False})
        st["ema"] = self.alpha * a.logit + (1 - self.alpha) * st["ema"]
        p = sigmoid(st["ema"])
        if p >= a.threshold:
            st["above"] += 1
            st["below"] = 0
        elif p < max(0.0, a.threshold - self.off_margin):
            st["below"] += 1
            st["above"] = 0
        was = st["active"]
        if not was and st["above"] >= self.on_frames:
            st["active"] = True
        elif was and st["below"] >= self.off_frames:
            st["active"] = False
        a.active = st["active"]
        a.onset = a.active and not was
        return a


def fit_weights(X: np.ndarray, y: np.ndarray, specs: List[CueSpec], prior_bias: float,
                l2: float = 1.0, iters: int = 2000, lr: float = 0.05, monotone: bool = True,
                sample_weight: Optional[np.ndarray] = None):
    """Fit additive weights on transformed cues, L2-pulled towards expert priors.

    ``monotone=True`` keeps every weight's sign equal to the expert prior's
    (projected gradient descent): more evidence can never *lower* the risk,
    which keeps the model safe and its explanations sensible even when the
    training labels have quirks.  Returns (bias, weights).
    """
    Z = np.stack([[s.transform(v) for s, v in zip(specs, row)] for row in X])
    w0 = np.array([s.weight for s in specs], dtype=float)
    w = w0.copy()
    b = float(prior_bias)
    n = len(y)
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    sw = sw * n / sw.sum()
    scale = Z.std(0) + 1e-6
    sign = np.sign(w0)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = (p - y) * sw
        gw = Z.T @ g / n + l2 * (w - w0) / n
        gb = g.mean()
        w -= lr * gw / scale**2
        b -= lr * gb
        if monotone:
            w = np.where(sign > 0, np.maximum(w, 0), np.where(sign < 0, np.minimum(w, 0), w))
    return b, w
