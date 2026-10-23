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
    "fall": ["fall", "falls", "fell", "fallen", "falling", "collapsed", "collapse", "collapsing", "tripped", "slipped"],
    "fire_smoke": ["smoke", "smoky", "fire", "fires", "flame", "flames", "burning", "blaze", "ablaze"],
    "crowd": ["crowd", "crowds", "crowding", "stampede", "panic", "panicking", "overcrowding", "overcrowded", "crush"],
}
CAUSAL_MARKERS = ("due to", "because", "caused by", "driven by", "triggered by")

NUM_RE = re.compile(r"(?<![\w#.\-/])[-+]?\d+(?:\.\d+)?%?")


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
    verify_text: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def explain(a: Assessment, model: HazardModel, location: str = "",
            recommendations: Optional[List[str]] = None, max_cues: int = 3,
            triage: Optional[dict] = None) -> Explanation:
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
    verify_text = ""
    if triage and triage.get("mode") == "VERIFY":
        verify_text = "Please verify before escalating: " + "; ".join(triage["reasons"]) + "."
    text = " ".join(x for x in [headline + ".", because, cf_text, conf, verify_text] if x)
    if recs:
        text += " Recommended: " + "; ".join(recs) + "."
    return Explanation(headline, because, cf_text, recs, conf, text, [c.name for c in top], verify_text)


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


def numbers_in(texts: Iterable[str]) -> List[float]:
    """Numbers appearing in trusted context strings (e.g. playbook actions, phone numbers)."""
    out: List[float] = []
    for t in texts:
        out += [float(n.rstrip("%")) for n in NUM_RE.findall(t)]
    return out


def verify(text: str, a: Assessment, model: HazardModel, location: str = "",
           extra_numbers: Iterable[float] = ()) -> Verification:
    """Check every hazard word, number and causal claim in ``text`` against the evidence.

    Rules (deliberately strict - a false accept is worse than a false reject,
    because a rejected narrator text simply falls back to the template):

    * hazard words: every hazard word in the narrative must belong to the
      assessed hazard, or be a keyword of a cue that actually raised the risk
      (e.g. "crowd" in "rapid exit movement (crowd flow index ...)" on a fire alert);
    * numbers in the narrative must be bound: next to a cue keyword -> equal to
      that cue's measured value (or its counterfactual); after "confidence /
      probability" -> the probability; after "threshold" -> the threshold.
      A number with no binding is accepted only if it equals the probability or
      the threshold - never "some cue's value" (that would allow swapping values
      between cues) and never a track id;
    * "N percent" is read as a percentage;
    * numbers in the trusted "Recommended:" section must equal numbers of the
      playbook text (``extra_numbers``) or a measured cue value;
    * in a sentence with a causal marker, every cue mentioned must have raised
      the risk.
    """
    v = Verification()
    # a cue is referred to by its phrase keywords or by its own name ("tilt rate" in counterfactuals)
    keywords: Dict[str, List[str]] = {name: [k.lower() for k in spec.keywords] + [name.replace("_", " ")]
                                      for name, spec in model.specs.items()}
    cue_by_name = {c.name: c for c in a.cues}
    cfs = counterfactuals(a, model)
    probs_ok: List[float] = [a.prob, a.threshold]
    # the location may contain digits (Room B-104): mask it so they are never read as claims
    masked = text
    if location:
        masked = re.sub(re.escape(location), "the zone", masked, flags=re.IGNORECASE)
    masked = re.sub(r"(\d+(?:\.\d+)?)\s*percent\b", r"\1%", masked, flags=re.IGNORECASE)
    parts = re.split(r"(?i)\brecommended:", masked, maxsplit=1)
    narrative, recs = parts[0], (parts[1] if len(parts) > 1 else "")

    # 1) hazard claims (narrative only; the playbook may mention other hazards on purpose)
    positive_kw = {k for c in a.cues if c.contribution > 0 for k in keywords.get(c.name, [])}
    low = narrative.lower()
    for hz, words in HAZARD_WORDS.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", low):
                ok = hz == a.hazard or w in positive_kw
                v.claims.append(Claim("hazard", w, ok, "" if ok else f"assessed hazard is {a.hazard}"))

    # 2) number + cause claims in the narrative, clause by clause; causality scoped to a sentence
    for sentence in _sentences(narrative):
        in_causal = False
        for clause in _clauses(sentence):
            if any(m in clause.lower() for m in CAUSAL_MARKERS):
                in_causal = True
            _check_clause(clause, in_causal, keywords, cue_by_name, cfs, probs_ok, a, v)

    # 3) numbers in the playbook section
    trusted = list(extra_numbers) + [c.value for c in a.cues]
    for mt in NUM_RE.finditer(recs):
        num = mt.group(0)
        ok = any(_matches(num, x) for x in trusted)
        v.claims.append(Claim("number", f"{num} [recommended]", ok, "" if ok else f"{num} not in the playbook or evidence"))
    return v


def _check_clause(clause: str, in_causal: bool, keywords: Dict[str, List[str]], cue_by_name: dict,
                  cfs: list, probs_ok: List[float], a: Assessment, v: Verification) -> None:
    cl = clause.lower()
    bound = [n for n, kws in keywords.items() if any(re.search(rf"\b{re.escape(k)}", cl) for k in kws)]
    for mt in NUM_RE.finditer(clause):
        num = mt.group(0)
        before = cl[max(0, mt.start() - 18):mt.start()]
        if re.search(r"threshold|alert line", before):
            ok = _matches(num, a.threshold)
            detail = "" if ok else f"{num} is not the alert threshold"
        elif re.search(r"confidence|probability|risk|likelihood|chance", before):
            ok = _matches(num, a.prob)
            detail = "" if ok else f"{num} is not the assessed probability"
        elif bound:
            cand = [cue_by_name[n].value for n in bound if n in cue_by_name]
            cand += [cf.needed for cf in cfs if cf.cue in bound]
            ok = any(_matches(num, x) for x in cand)
            detail = "" if ok else f"{num} does not match measured {bound}"
        else:
            ok = any(_matches(num, g) for g in probs_ok) and num.endswith("%")
            detail = "" if ok else f"{num} is not bound to any measured cue"
        v.claims.append(Claim("number", f"{num} [{clause[:60]}]", ok, detail))
    if in_causal:
        for n in bound:
            c = cue_by_name.get(n)
            ok = c is not None and c.contribution > 0
            v.claims.append(Claim("cause", n, ok, "" if ok else "cited as a cause but did not raise the risk"))
