"""Calibration and conformal false-alarm control for alert thresholds.

* :func:`platt_fit` / :func:`platt_apply` - temperature + shift on the logit.
* :func:`ece` - expected calibration error (equal-width bins).
* :func:`conformal_threshold` - split-conformal threshold on *normal* data so
  that, for a new normal window exchangeable with the calibration set,
  P(score > tau) <= alpha.  This turns "how many false alarms will this
  camera raise?" into a stated, testable guarantee instead of a hand-tuned
  constant.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def platt_fit(logits: np.ndarray, y: np.ndarray, iters: int = 500, lr: float = 0.1) -> Tuple[float, float]:
    """Fit p = sigmoid(a * logit + b) by gradient descent on log-loss."""
    logits = np.asarray(logits, float)
    y = np.asarray(y, float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * logits + b)))
        g = p - y
        a -= lr * float(np.mean(g * logits))
        b -= lr * float(np.mean(g))
    return a, b


def platt_apply(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(a * np.asarray(logits, float) + b)))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs, float)
    y = np.asarray(y, float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & ((probs < hi) if hi < 1 else (probs <= hi))
        if m.any():
            total += m.mean() * abs(probs[m].mean() - y[m].mean())
    return float(total)


def conformal_threshold(normal_scores: np.ndarray, alpha: float) -> float:
    """Split-conformal upper quantile of scores on normal (negative) data.

    Uses the finite-sample corrected rank ceil((n+1)(1-alpha)).  If that rank
    exceeds n the guarantee cannot be met with this many samples and +inf is
    returned (never alert) - callers should collect more calibration data.
    """
    s = np.sort(np.asarray(normal_scores, float))
    n = len(s)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf
    return float(s[k - 1])


def logit_to_prob_threshold(tau_logit: float) -> float:
    if math.isinf(tau_logit):
        return 1.0
    return 1 / (1 + math.exp(-tau_logit))
