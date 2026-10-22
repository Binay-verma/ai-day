"""Operator Q&A over an alert's evidence record ("Ask the copilot").

Deterministic intent matching, answering only from the stored evidence
(cues, contributions, counterfactuals, recommendations, verification), so
answers are as grounded as the explanation itself and work offline on the
edge box.  Typical questions:

* "why?" / "what caused this?"
* "what if the sensor was wrong?" / "would it fire without smoke?"
* "how sure are you?"  * "what should I do?"  * "is this a false alarm?"
"""
from __future__ import annotations

import math
import re
from typing import Dict, List

INTENTS = [
    ("whatif", r"\b(what if|without|ignore|would it|wouldn'?t|had not|hadn'?t|if the)\b"),
    ("false", r"\b(false alarm|false positive|sure it'?s real|really|mistake)\b"),
    ("confidence", r"\b(how sure|confiden|certain|probab|likely)\b"),
    ("action", r"\b(what should|what do|do now|next|action|recommend|steps?)\b"),
    ("evidence", r"\b(evidence|sensor|camera|see|saw|heatmap|where)\b"),
    ("why", r"\b(why|cause|reason|explain|because)\b"),
]


def _name(c: str) -> str:
    return c.replace("_", " ")


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sig(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def answer(alert: Dict, question: str) -> Dict:
    q = question.lower().strip()
    intent = next((name for name, pat in INTENTS if re.search(pat, q)), "why")
    cues: List[Dict] = alert.get("cues", [])
    pos = [c for c in cues if c["contribution"] > 0]
    neg = [c for c in cues if c["contribution"] < 0]
    dec = alert.get("decision", {})
    prob = alert["prob"]
    thr = alert["explanation"].get("confidence_text", "")
    if intent == "why":
        parts = [f"{c['phrase']} (x{math.exp(c['contribution']):.1f} odds)" for c in pos[:4]]
        txt = "The risk score rose because of " + "; ".join(parts) + "." if parts else "No cue raised the risk strongly."
        if neg:
            txt += " Counting against it: " + "; ".join(c["phrase"] for c in neg[:2]) + "."
        return {"intent": intent, "answer": txt}
    if intent == "whatif":
        # find a cue mentioned in the question and remove it (set to reference)
        mentioned = [c for c in cues if any(w in q for w in c["name"].split("_") if len(w) > 2)]
        if not mentioned and re.search(r"sensor", q):
            mentioned = [c for c in cues if c.get("source") == "sensor"]
        if mentioned:
            delta = sum(c["contribution"] for c in mentioned)
            new_p = _sig(_logit(prob) - delta)
            names = ", ".join(_name(c["name"]) for c in mentioned)
            still = "would still" if new_p >= _threshold(alert) else "would NOT"
            return {"intent": intent, "answer": f"Without {names}, the risk would be {new_p:.0%} instead of {prob:.0%}, so the alert {still} have fired."}
        cfs = dec.get("counterfactuals") or []
        if cfs:
            return {"intent": intent, "answer": "It would not have fired if " + "; or if ".join(cf["text"] for cf in cfs[:3]) + "."}
        suff = dec.get("sufficient_reason")
        return {"intent": intent, "answer": f"No single cue flips the decision; the combination {', '.join(map(_name, suff or []))} is needed."}
    if intent == "confidence":
        margin = dec.get("margin_logit", 0.0)
        return {"intent": intent, "answer": f"Calibrated risk is {prob:.0%}; the score is {margin:.1f} log-odds above the alert threshold. {thr}"}
    if intent == "action":
        acts = alert.get("recommendations", [])
        if not acts:
            return {"intent": intent, "answer": "No playbook step applies; verify on camera."}
        return {"intent": intent, "answer": " ".join(f"{i + 1}. {a['text']}." for i, a in enumerate(acts))}
    if intent == "false":
        ver = alert.get("verification", {})
        weakest = min(pos, key=lambda c: c["contribution"]) if pos else None
        txt = (f"Possible. The alert rests mainly on {_name(pos[0]['name'])}" if pos else "Possible.")
        if weakest and len(pos) > 1:
            txt += f"; the weakest supporting cue is {_name(weakest['name'])}"
        txt += f". Explanation groundedness {ver.get('groundedness', 1):.0%}. If it is a false alarm, press Dismiss: the zone's threshold adapts (bounded)."
        return {"intent": intent, "answer": txt}
    # evidence
    src = {}
    for c in cues:
        src.setdefault(c.get("source", "vision"), []).append(c["phrase"])
    txt = " ".join(f"{k.capitalize()}: {'; '.join(v)}." for k, v in src.items())
    return {"intent": "evidence", "answer": txt + " The evidence image highlights the regions the model relied on."}


def _threshold(alert: Dict) -> float:
    m = re.search(r"threshold (\d+)%", alert["explanation"].get("confidence_text", ""))
    return int(m.group(1)) / 100 if m else 0.5
