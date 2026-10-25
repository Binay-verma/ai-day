"""Natural-language search over the incident store (idea ported from the team's Argus project,
re-implemented for this copilot's hazards).

"falls in Corridor B last night", "how many fire alerts this week", "open alerts in the last 2 hours",
"false alarms by camera yesterday" ... The question is parsed by a deterministic keyword/date parser into
an ALLOW-LISTED filter (hazards, zones, cameras, statuses, severities, triage, time window, aggregation).
Zones and cameras must be names that exist in this site's config. Nothing is ever turned into SQL: rows
come from the store's parameterised query and are filtered in Python. The answer cites alert ids, and
shows what was understood as editable chips, so the operator can see exactly what was searched.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

HAZARD_WORDS = {
    "fall": r"falls?|fell|fallen|falling|person down|people down|collapsed?|on the (?:floor|ground)|long[- ]lie|still down",
    "fire_smoke": r"fires?|smoke|smoky|flames?|burning",
    "crowd": r"crowds?|crowding|overcrowd\w*|stampede|panic|dispersal|rush(?:ing)?",
    "intrusion": r"intrusions?|trespass\w*|restricted(?: area)?",
    "camera_fault": r"camera (?:faults?|problems?|trouble|issues?)|tamper\w*|blackouts?|lens covered|camera moved",
    "exit_blocked": r"exits? (?:blocked|unsafe|alerts?)|blocked exits?|unsafe exits?",
    "crowd_toward_hazard": r"heading (?:into|towards?) (?:danger|the fire|smoke|hazards?)|crowd towards? (?:danger|hazards?)",
}
HAZARD_LABEL = {"fall": "falls", "fire_smoke": "fire / smoke", "crowd": "crowd", "intrusion": "intrusion",
                "camera_fault": "camera trouble", "self_test": "self-test", "exit_blocked": "unsafe exits",
                "crowd_toward_hazard": "crowd heading into danger"}
STATUS_WORDS = [("open", r"open|unacknowledged|unhandled|pending|unreviewed|not (?:yet )?(?:handled|reviewed)|active"),
                ("confirmed", r"confirmed|real|genuine|true alarms?"),
                ("dismissed", r"dismissed|false alarms?|false positives?|rejected"),
                ("resolved", r"resolved|cleared")]
SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
PARTS = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 22), "night": (22, 30)}  # night ends 06:00
MAX_DAYS = 90
MAX_LIST = 25


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9:'\- ]+", " ", s.lower())).strip()


def _midnight(d: dt.date) -> float:
    return dt.datetime(d.year, d.month, d.day).timestamp()


def _find_names(text: str, names: Iterable[str]) -> List[str]:
    """Longest-first whole-word matches of configured names (case-insensitive)."""
    found: List[str] = []
    taken: List[Tuple[int, int]] = []
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        for m in re.finditer(rf"(?<![a-z0-9]){re.escape(_norm(name))}(?![a-z0-9])", text):
            if not any(m.start() < b and a < m.end() for a, b in taken):
                taken.append((m.start(), m.end()))
                found.append(name)
                break
    return found


def parse(question: str, zones: Iterable[str], cameras: Iterable[str], now: Optional[float] = None) -> Dict[str, Any]:
    """Question -> allow-listed filter plus the human-readable chips of what was understood."""
    now = dt.datetime.now().timestamp() if now is None else now
    s = _norm(question)
    q: Dict[str, Any] = {"hazards": [], "zones": [], "cameras": [], "statuses": [], "severities": [], "triage": [],
                         "since": None, "until": None, "aggregation": "list"}
    chips: List[str] = []
    # cameras before zones so "cam1" inside a zone name cannot be taken twice
    q["cameras"] = _find_names(s, cameras)
    q["zones"] = [z for z in _find_names(s, zones) if z not in q["cameras"]]
    q["hazards"] = [h for h, pat in HAZARD_WORDS.items() if re.search(rf"\b(?:{pat})\b", s)]
    if re.search(r"\blong[- ]lie|still down\b", s):
        q["severities"].append("CRITICAL")
    for st, pat in STATUS_WORDS:
        if re.search(rf"\b(?:{pat})\b", s):
            q["statuses"].append(st)
    for sev in SEVERITIES:
        if re.search(rf"\b{sev.lower()}\b", s) and sev not in q["severities"]:
            q["severities"].append(sev)
    if re.search(r"\burgent|serious|high[- ]priority\b", s):
        q["severities"] = sorted(set(q["severities"]) | {"CRITICAL", "HIGH"}, key=SEVERITIES.index)
    if re.search(r"\bverify\b|\bneeds? (?:checking|verification)\b", s):
        q["triage"].append("VERIFY")
    if re.search(r"\balarm mode\b|\bALARM\b", question):
        q["triage"].append("ALARM")
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal\b", s):
        q["aggregation"] = "count"
    if re.search(r"\b(?:by|per|each|which) (?:zones?|areas?|places?)\b|\bwhere\b", s):
        q["aggregation"] = "by_zone"
    elif re.search(r"\b(?:by|per|each|which) cameras?\b", s):
        q["aggregation"] = "by_camera"
    elif re.search(r"\b(?:by|per|each) hour\b|\bhourly\b|\bwhat times?\b|\bwhen\b|\bbusiest\b", s):
        q["aggregation"] = "by_hour"
    elif re.search(r"\b(?:by|per|each|which) (?:hazards?|types?|kinds?)\b|\bbreakdown\b", s):
        q["aggregation"] = "by_hazard"
    # time window (server local time)
    today = dt.datetime.fromtimestamp(now).date()
    m = re.search(r"\b(?:last|past|previous|within the last|in the last)\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|h|days?|weeks?)\b", s)
    part = next((p for p in PARTS if re.search(rf"\b{p}\b", s)), None) or ("night" if "tonight" in s else None)
    if m:
        n, unit = float(m.group(1)), m.group(2)
        secs = n * (60 if unit.startswith("m") else 86400 if unit.startswith("d") else 604800 if unit.startswith("w") else 3600)
        q["since"] = now - min(secs, MAX_DAYS * 86400)
        chips.append(f"last {m.group(1)} {unit}")
    elif re.search(r"\b(?:last|past) hour\b", s):
        q["since"] = now - 3600
        chips.append("last hour")
    elif re.search(r"\blast night\b|\bovernight\b|\byesterday night\b", s):
        y = today - dt.timedelta(days=1)
        q["since"], q["until"] = _midnight(y) + 22 * 3600, _midnight(today) + 6 * 3600
        chips.append("last night (22:00-06:00)")
    elif re.search(r"\byesterday\b", s):
        y = today - dt.timedelta(days=1)
        lo, hi = PARTS[part] if part and part != "night" else (0, 24)
        q["since"], q["until"] = _midnight(y) + lo * 3600, _midnight(y) + hi * 3600
        chips.append(f"yesterday{' ' + part if part and part != 'night' else ''}")
    elif re.search(r"\btoday\b|\bthis (?:morning|afternoon|evening)\b|\btonight\b", s):
        lo, hi = PARTS[part] if part else (0, 24)
        q["since"], q["until"] = _midnight(today) + lo * 3600, min(now, _midnight(today) + hi * 3600)
        chips.append(f"today{' ' + part if part else ''}")
    elif re.search(r"\bthis week\b", s):
        q["since"] = _midnight(today - dt.timedelta(days=today.weekday()))
        chips.append("this week")
    elif re.search(r"\b(?:last|past) (?:week|7 days)\b", s):
        q["since"] = now - 7 * 86400
        chips.append("last 7 days")
    elif re.search(r"\b(?:this|last|past) month\b|\b30 days\b", s):
        q["since"] = now - 30 * 86400
        chips.append("last 30 days")
    else:
        q["since"] = now - 24 * 3600
        chips.append("last 24 hours (default)")
    chips = ([HAZARD_LABEL.get(h, h) for h in q["hazards"]] + q["zones"] + [f"camera {c}" for c in q["cameras"]]
             + q["statuses"] + [s_.lower() for s_ in q["severities"]] + q["triage"] + chips
             + ([q["aggregation"].replace("_", " ")] if q["aggregation"] != "list" else []))
    q["understood"] = chips
    return q


def run(store, q: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    rows = store.list_alerts(5000, since=q.get("since"))

    def keep(a: Dict[str, Any]) -> bool:
        if q.get("until") is not None and (a.get("wall_time") or 0) > q["until"]:
            return False
        if a.get("hazard") == "self_test" and "camera_fault" not in q["hazards"]:
            return False
        tri = a.get("triage")
        tri = tri.get("mode") if isinstance(tri, dict) else tri
        return ((not q["hazards"] or a.get("hazard") in q["hazards"]) and (not q["zones"] or a.get("zone") in q["zones"])
                and (not q["cameras"] or a.get("camera_id") in q["cameras"])
                and (not q["statuses"] or a.get("status") in q["statuses"])
                and (not q["severities"] or a.get("severity") in q["severities"])
                and (not q["triage"] or tri in q["triage"]))
    hits = [a for a in rows if keep(a)]
    agg = q.get("aggregation", "list")
    groups: List[Tuple[str, int]] = []
    if agg in ("by_zone", "by_camera", "by_hazard", "by_hour"):
        def key(a: Dict[str, Any]) -> str:
            if agg == "by_hour":
                return dt.datetime.fromtimestamp(a.get("wall_time") or 0).strftime("%H:00")
            v = a.get({"by_zone": "zone", "by_camera": "camera_id", "by_hazard": "hazard"}[agg]) or "?"
            return HAZARD_LABEL.get(v, v) if agg == "by_hazard" else v
        groups = Counter(key(a) for a in hits).most_common()
    cited = [a["id"] for a in hits[:MAX_LIST]]
    what = ", ".join(q["understood"])
    if not hits:
        summary = f"No alerts match ({what})."
    elif agg == "count":
        summary = f"{len(hits)} alert{'s' if len(hits) != 1 else ''} match ({what})."
    elif groups:
        top = "; ".join(f"{k}: {n}" for k, n in groups[:6])
        summary = f"{len(hits)} alerts ({what}). {top}."
    else:
        summary = f"{len(hits)} alert{'s' if len(hits) != 1 else ''} ({what}), newest first."
    if cited:
        summary += " Cited: " + ", ".join(cited[:5]) + (" ..." if len(cited) > 5 else "")
    slim = [{k: a.get(k) for k in ("id", "hazard", "zone", "camera_id", "severity", "status", "wall_time", "peak_prob")}
            | {"headline": (a.get("explanation") or {}).get("headline", "")} for a in hits[:MAX_LIST]]
    return {"query": {k: v for k, v in q.items()}, "understood": q["understood"], "total": len(hits),
            "groups": [{"key": k, "count": n} for k, n in groups], "alerts": slim, "summary": summary,
            "note": "Deterministic search of the incident store; every number comes from the listed alerts."}
