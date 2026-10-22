"""Grounded natural-language explanations + an automatic grounding verifier.

Two parts:

1. :func:`explain` turns an :class:`Assessment` (+ counterfactuals and
   recommendations) into an operator-facing explanation using deterministic
   templates.  Every number printed comes from the evidence record, so the
   text is grounded by construction.

2. :func:`verify` checks *any* explanation text (template output, an LLM
   paraphrase, or an operator's note) against the evidence record:
   every number must be bound to a cue that actually has that value, every
   cue cited as a *cause* must have a positive contribution, and the hazard
   named must be the hazard assessed.  This is what lets an optional
   on-device LLM be used safely: its output is accepted only if it verifies,
   otherwise the template text is used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .counterfactual import counterfactuals, sufficient_reason
from .fusion import Assessment, HazardModel

HAZARD_TITLES = {
    "fall": "Fall detected",
    "fire_smoke": "Smoke / fire detected",
    "crowd": "Abnormal crowd behaviour",
}
HAZARD_WORDS = {
    "fall": ["fall", "fallen", "collapsed", "collapse"],
    "fire_smoke": ["smoke", "fire", "flame", "flames", "burning"],
    "crowd": ["crowd", "crowding", "stampede", "panic", "overcrowding"],
}
CAUSAL_MARKERS = ("due to", "because", "caused by", "driven by", "triggered by")

NUM_RE = re.compile(r"(?<![\w#])[-+]?\d+(?:\.\d+)?%?")


def _join(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


@dataclass
class Explanation:
    headline: str
    because: str
    counterfactual: str
    recommendations: List[str]
    confidence_text: str
    text: str
    cited_cues: List[str]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def explain(a: Assessment, model: HazardModel, location: str = "",
            recommendations: Optional[List[str]] = None, max_cues: int = 3) -> Explanation:
    title = HAZARD_TITLES.get(a.hazard, a.hazard)
    where = f" in {location}" if location else ""
    who = f" ({a.subject})" if a.subject and a.subject.startswith("person") else ""
    headline = f"{title}{who}{where}"
    top = a.top_cues(max_cues)
    reasons = [c.phrase for c in top]
    because = f"Alert raised due to {_join(reasons)}." if reasons else "Alert raised."
    cfs = counterfactuals(a, model)
    cf_text = ""
    if cfs:
        cf_text = f"It would not have fired if {cfs[0].text()}."
    else:
        suff = sufficient_reason(a)
        if suff and len(suff) > 1:
            cf_text = ("No single cue explains it: the combination of "
                       f"{_join([s.replace('_', ' ') for s in suff])} was needed.")
    pct = round(a.prob * 100)
    conf = f"Confidence {pct}% (threshold {round(a.threshold * 100)}%)."
    recs = recommendations or []
    text = " ".join(x for x in [headline + ".", because, cf_text, conf] if x)
    if recs:
        text += " Recommended: " + "; ".join(recs) + "."
    return Explanation(headline, because, cf_text, recs, conf, text, [c.name for c in top])


# ----------------------------------------------------------------- verifier
@dataclass
class Claim:
    kind: str  # number | cause | hazard
    text: str
    supported: bool
    detail: str = ""


@dataclass
class Verification:
    claims: List[Claim] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.claims)

    @property
    def unsupported(self) -> List[Claim]:
        return [c for c in self.claims if not c.supported]

    @property
    def groundedness(self) -> float:
        return 1.0 if not self.claims else sum(c.supported for c in self.claims) / len(self.claims)

    @property
    def ok(self) -> bool:
        return not self.unsupported

    def to_dict(self) -> dict:
        return {"groundedness": self.groundedness, "n_claims": self.n,
                "unsupported": [c.__dict__ for c in self.unsupported]}


def _matches(printed: str, value: float) -> bool:
    """Does the printed number plausibly render ``value`` (any rounding)?

    ``"70%"`` matches 0.70 (fractions printed as percentages) and 70.
    """
    pct = printed.endswith("%")
    core = printed.rstrip("%")
    try:
        x = float(core)
    except ValueError:
        return False
    decimals = len(core.split(".")[1]) if "." in core else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    cands = [value, value * 100] if pct else [value]
    return any(abs(x - c) <= max(tol, 0.02 * abs(c)) for c in cands)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.;!?])\s+", text) if s.strip()]


def _clauses(sentence: str) -> List[str]:
    return [c.strip() for c in re.split(r",\s+|\band\b|\bwith\b", sentence) if c.strip()]


def verify(text: str, a: Assessment, model: HazardModel, location: str = "",
           extra_numbers: Iterable[float] = ()) -> Verification:
    v = Verification()
    low = text.lower()
    keywords: Dict[str, List[str]] = {name: [k.lower() for k in spec.keywords]
                                      for name, spec in model.specs.items()}
    cue_by_name = {c.name: c for c in a.cues}
    # numbers that are legitimately printable besides cue values
    cfs = counterfactuals(a, model)
    globals_ok: List[float] = [a.prob * 100, a.threshold * 100, a.prob, a.threshold]
    globals_ok += [cf.needed for cf in cfs] + list(extra_numbers)
    sid = re.findall(r"\d+", a.subject or "")
    globals_ok += [float(s) for s in sid] + [float(s) for s in re.findall(r"\d+", location)]

    # 1) hazard claims: any hazard word mentioned must be the assessed hazard
    for hz, words in HAZARD_WORDS.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", low):
                ok = hz == a.hazard or any(
                    w in keywords.get(c.name, []) for c in a.cues)
                v.claims.append(Claim("hazard", w, ok, "" if ok else f"assessed hazard is {a.hazard}"))
                break

    # 2) number + cause claims, clause by clause; causality is scoped to a sentence
    for sentence in _sentences(text):
        in_causal = False
        for clause in _clauses(sentence):
            if any(m in clause.lower() for m in CAUSAL_MARKERS):
                in_causal = True
            _check_clause(clause, in_causal, keywords, cue_by_name, cfs, globals_ok, a, v)
    return v


def _check_clause(clause: str, in_causal: bool, keywords: Dict[str, List[str]], cue_by_name: dict,
                  cfs: list, globals_ok: List[float], a: Assessment, v: Verification) -> None:
    cl = clause.lower()
    bound = [n for n, kws in keywords.items() if any(re.search(rf"\b{re.escape(k)}", cl) for k in kws)]
    for num in NUM_RE.findall(clause):
        if bound:
            cand = [cue_by_name[n].value for n in bound if n in cue_by_name]
            cand += [cf.needed for cf in cfs if cf.cue in bound]
            ok = any(_matches(num, x) for x in cand) if cand else any(_matches(num, g) for g in globals_ok)
            detail = "" if ok else f"{num} does not match measured {bound}"
        else:
            ok = any(_matches(num, g) for g in globals_ok) or any(_matches(num, c.value) for c in a.cues)
            detail = "" if ok else f"{num} not found in evidence"
        v.claims.append(Claim("number", f"{num} [{clause[:60]}]", ok, detail))
    if in_causal:
        for n in bound:
            c = cue_by_name.get(n)
            ok = c is not None and c.contribution > 0
            v.claims.append(Claim("cause", n, ok, "" if ok else "cited as a cause but did not raise the risk"))
