import math

import numpy as np
import pytest

from edgecopilot.copilot.qa import answer
from edgecopilot.copilot.recommender import Recommender, safe_eval
from edgecopilot.pipeline import load_config
from edgecopilot.xai.calibration import conformal_threshold, ece, platt_apply, platt_fit
from edgecopilot.xai.counterfactual import counterfactuals, necessary_cues, sufficient_reason
from edgecopilot.xai.fusion import HazardModel, Hysteresis, sigmoid
from edgecopilot.xai.nlg import explain, verify

CFG = load_config("configs/campus.yaml")


def fall_model():
    return HazardModel.from_config("fall", CFG["hazards"]["fall"])


FALL = {"torso_angle": 82, "descent_speed": 1.8, "height_drop": 0.62, "aspect_ratio": 1.9, "ground_time": 3.0}
STAND = {"torso_angle": 6, "descent_speed": 0.05, "height_drop": 0.0, "aspect_ratio": 0.4, "ground_time": 0.0}


def test_attribution_is_exact():
    m = fall_model()
    rng = np.random.default_rng(0)
    for _ in range(200):
        vals = {k: float(rng.uniform(s.lo, s.hi)) for k, s in m.specs.items()}
        a = m.assess("person#1", 0.0, vals, {"torso_angle": float(rng.uniform())})
        assert a.attribution_error() < 1e-9
        assert math.isclose(a.prob, sigmoid(a.logit))


def test_fall_vs_standing():
    m = fall_model()
    assert m.assess("p", 0, FALL).prob > 0.9
    assert m.assess("p", 0, STAND).prob < 0.01


def test_counterfactual_flips_decision():
    m = fall_model()
    a = m.assess("person#1", 0, FALL)
    cfs = counterfactuals(a, m)
    assert cfs, "an active alert should have at least one single-cue counterfactual"
    for cf in cfs:
        vals = dict(FALL)
        vals[cf.cue] = cf.needed
        b = m.assess("person#1", 0, vals)
        assert b.logit < b.logit_threshold, cf


def test_sufficient_and_necessary():
    m = fall_model()
    a = m.assess("person#1", 0, FALL)
    s = sufficient_reason(a)
    only = {k: (FALL[k] if k in s else m.specs[k].ref) for k in FALL}
    assert m.assess("p", 0, only).logit >= a.logit_threshold
    for n in necessary_cues(a):
        vals = dict(FALL)
        vals[n] = m.specs[n].ref
        assert m.assess("p", 0, vals).logit < a.logit_threshold


def test_template_explanation_is_fully_grounded():
    m = fall_model()
    a = m.assess("person#3", 0, FALL)
    e = explain(a, m, "Corridor B", ["Send first-aid to Corridor B"])
    v = verify(e.text, a, m, "Corridor B")
    assert v.ok, v.to_dict()
    assert "torso" in e.because


@pytest.mark.parametrize("bad,good", [("82 deg", "45 deg"), ("1.8 body", "2.9 body")])
def test_verifier_catches_wrong_numbers(bad, good):
    m = fall_model()
    a = m.assess("person#3", 0, FALL)
    e = explain(a, m, "Corridor B")
    assert bad in e.text
    v = verify(e.text.replace(bad, good), a, m, "Corridor B")
    assert not v.ok


def test_verifier_catches_wrong_hazard_and_false_cause():
    m = fall_model()
    a = m.assess("person#3", 0, dict(FALL, ground_time=0.0))
    assert not verify("Smoke detected near person 3.", a, m).ok
    # ground_time=0 lowers the risk -> citing it as a cause must fail
    assert not verify("Alert raised due to no movement on the ground for 0.0 s.", a, m).ok


def test_hysteresis_needs_consecutive_frames():
    m = fall_model()
    h = Hysteresis(alpha=1.0, on_frames=2, off_frames=2)
    hi, lo = m.assess("p", 0, FALL), m.assess("p", 0, STAND)
    assert not h.update("k", hi).active
    assert h.update("k", m.assess("p", 0, FALL)).onset
    assert h.update("k", m.assess("p", 0, FALL)).active
    h.update("k", lo)
    assert not h.update("k", m.assess("p", 0, STAND)).active


def test_feedback_is_bounded():
    m = fall_model()
    for _ in range(100):
        m.feedback("zone", 0, 0.99, lr=0.5, max_shift=2.0)
    assert m.bias_offset["zone"] == -2.0


def test_conformal_threshold_controls_false_alarms():
    rng = np.random.default_rng(1)
    alpha = 0.05
    rates = []
    for _ in range(200):
        cal, test = rng.normal(size=500), rng.normal(size=2000)
        tau = conformal_threshold(cal, alpha)
        rates.append((test > tau).mean())
    assert np.mean(rates) <= alpha + 0.005
    assert conformal_threshold(np.ones(5), 0.01) == math.inf


def test_platt_and_ece():
    rng = np.random.default_rng(2)
    z = rng.normal(size=4000) * 3
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-z / 3))).astype(float)
    a, b = platt_fit(z, y)
    assert abs(a - 1 / 3) < 0.08
    assert ece(platt_apply(z, a, b), y) < ece(1 / (1 + np.exp(-z)), y)


def test_safe_eval_blocks_code():
    assert safe_eval("ground_time > 8 and prob >= 0.5", {"ground_time": 9, "prob": 0.6})
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('true')", {})


def test_recommender_and_qa():
    m = fall_model()
    a = m.assess("person#3", 0, dict(FALL, ground_time=20))
    rec = Recommender.from_config(CFG).recommend(a, "Corridor B")
    texts = " ".join(x["text"] for x in rec["actions"])
    assert "AED outside Room B-104" in texts and "108" in texts
    assert rec["severity"] == "CRITICAL"
    e = explain(a, m, "Corridor B", [x["text"] for x in rec["actions"]])
    from edgecopilot.xai.counterfactual import explain_decision
    alert = {"cues": [c.to_dict() for c in a.cues], "decision": explain_decision(a, m), "prob": a.prob,
             "explanation": e.to_dict(), "recommendations": rec["actions"], "verification": {"groundedness": 1.0}}
    assert "torso" in answer(alert, "why did this fire?")["answer"]
    r = answer(alert, "what if the torso angle was wrong?")
    assert r["intent"] == "whatif" and "Without torso angle" in r["answer"]
    assert answer(alert, "what should I do now?")["intent"] == "action"
