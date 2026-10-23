"""SOP-playbook recommendations grounded in the campus zone map.

Recommendations are not free text from a model: they are instantiated from
playbooks written by the campus safety team (``configs/campus.yaml``) and
filled with facts from the zone map (nearest exits, AED, extinguisher,
assembly point, duty contacts) and from the evidence (e.g. a fall with a
long ground time escalates to an ambulance call).  Each action keeps a
``because`` link to the cue/condition that triggered it, so the operator can
see why the copilot suggests it.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..xai.fusion import Assessment


_OPS = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Eq: operator.eq, ast.NotEq: operator.ne}


def safe_eval(expr: str, env: Dict[str, float]) -> bool:
    """Evaluate a playbook condition like ``ground_time > 10 and prob >= 0.8``.

    Only names, numbers, comparisons, and/or/not are allowed - no calls,
    attributes or imports - so a config file can never execute code.
    """
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BoolOp):
            vals = [ev(v) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not ev(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                if type(op) not in _OPS or not _OPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise NameError(node.id)
            return env[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"disallowed expression element: {type(node).__name__}")
    return bool(ev(ast.parse(expr, mode="eval")))


@dataclass
class Zone:
    name: str
    capacity: int = 50
    exits: List[str] = field(default_factory=list)
    aed: str = ""
    extinguisher: str = ""
    first_aid: str = ""
    assembly_point: str = ""
    warden: str = ""


@dataclass
class Action:
    text: str
    priority: int  # 1 = do now
    because: str


SEVERITY = [(0.9, "CRITICAL"), (0.75, "HIGH"), (0.0, "MEDIUM")]


def severity(a: Assessment, escalate: bool = False) -> str:
    for th, name in SEVERITY:
        if a.prob >= th:
            if escalate and name != "CRITICAL":
                return "CRITICAL" if name == "HIGH" else "HIGH"
            return name
    return "LOW"


class Recommender:
    def __init__(self, zones: Dict[str, Zone], playbooks: Dict[str, List[dict]], contacts: Optional[dict] = None):
        self.zones = zones
        self.playbooks = playbooks
        self.contacts = contacts or {}

    @classmethod
    def from_config(cls, cfg: dict) -> Recommender:
        zones = {name: Zone(name=name, **z) for name, z in cfg.get("zones", {}).items()}
        return cls(zones, cfg.get("playbooks", {}), cfg.get("contacts", {}))

    def _cond(self, cond: Optional[str], vals: Dict[str, float], a: Assessment) -> bool:
        if not cond:
            return True
        env = dict(vals)
        env.update({"prob": a.prob})
        try:
            return safe_eval(cond, env)
        except NameError:
            return False  # condition refers to a cue that was not measured

    def recommend(self, a: Assessment, zone_name: str) -> Dict:
        z = self.zones.get(zone_name) or Zone(name=zone_name)
        vals = {c.name: c.value for c in a.cues}
        fmt = {
            "zone": z.name, "exits": " / ".join(z.exits) or "the nearest exit",
            "exit": z.exits[0] if z.exits else "the nearest exit",
            "aed": z.aed or "the nearest AED", "extinguisher": z.extinguisher or "the nearest extinguisher",
            "first_aid": z.first_aid or "the first-aid team", "assembly_point": z.assembly_point or "the assembly point",
            "warden": z.warden or "the zone warden", "subject": a.subject,
            **{k: v for k, v in self.contacts.items()}, **{k: round(v, 1) for k, v in vals.items()},
        }
        actions: List[Action] = []
        escalate = False
        for step in self.playbooks.get(a.hazard, []):
            if self._cond(step.get("when"), vals, a):
                try:
                    text = step["do"].format(**fmt)
                except KeyError:
                    text = step["do"]
                actions.append(Action(text, int(step.get("priority", 2)), step.get("when") or "standard procedure"))
                escalate |= bool(step.get("escalate", False))
        actions.sort(key=lambda x: x.priority)
        return {"severity": severity(a, escalate), "actions": [a_.__dict__ for a_ in actions]}
