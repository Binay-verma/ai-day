"""Site awareness: cameras working together around exits and hazards (no identity, no tracking of people).

Two site-level features build on the per-camera results:

1. **Safe-exit guidance.** Each exit named in the zone map can be watched by cameras
   (``site.exits.<name>.cameras``) or lie in a zone of its own. When smoke or fire reaches the alert line
   on a camera watching an exit, that exit is marked unsafe (and stays so for ``hazard_ttl_s`` after the
   hazard clears, so guidance does not flip-flop). Every recommendation then routes people to the exits
   that are still safe and says which exit to avoid and why; an ``exit_blocked`` advisory is raised once
   per exit so evacuation can be redirected.
2. **Crowd heading into danger.** Each camera reports where its moving people are heading (image edges
   mapped to places by ``cameras[].edges``; the heading predictor is measured in RESULTS section 16). When
   at least ``min_people`` are heading for an exit or zone where fire/smoke is active, a
   ``crowd_toward_hazard`` alert names the place, the evidence and the safe alternative.

All reasons are built from the live per-camera gauges (score, alert line, camera), so every sentence of
the advice can be traced to a measurement.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ExitInfo:
    name: str
    zone: str = ""                         # zone the exit itself is in (if it is a zone)
    cameras: List[str] = field(default_factory=list)


@dataclass
class Hazard:
    camera: str
    zone: str
    hazard: str
    prob: float
    threshold: float
    t: float


class SiteState:
    def __init__(self, cfg: dict, camera_zones: Optional[Dict[str, str]] = None):
        site = cfg.get("site") or {}
        self.zones: Dict[str, dict] = cfg.get("zones", {}) or {}
        self.exits: Dict[str, ExitInfo] = {n: ExitInfo(n, e.get("zone", "") or "", list(e.get("cameras", []) or []))
                                           for n, e in (site.get("exits") or {}).items()}
        self.ttl_s = float(site.get("hazard_ttl_s", 120.0))
        ct = site.get("crowd_toward_hazard") or {}
        self.min_people = int(ct.get("min_people", 3))
        self.cooldown_s = float(ct.get("cooldown_s", 60.0))
        self.max_eta_s = float(ct.get("max_eta_s", 20.0))
        self.camera_zones = dict(camera_zones or {})
        self._lock = threading.Lock()
        self._hazards: Dict[Tuple[str, str], Hazard] = {}    # (camera, hazard) -> latest at/above the alert line
        self._unsafe_since: Dict[str, float] = {}
        self._last_crowd_alert: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------ inputs
    def observe(self, camera: str, gauges: Dict[str, dict], now: Optional[float] = None) -> List[dict]:
        """Feed one camera's live gauges ({hazard: {prob, threshold, active}}). Returns exit transitions."""
        now = time.time() if now is None else now
        zone = self.camera_zones.get(camera, "")
        with self._lock:
            for hz in ("fire_smoke", "crowd"):
                g = gauges.get(hz)
                if g and (g.get("active") or g.get("prob", 0.0) >= g.get("threshold", 1.0)):
                    self._hazards[(camera, hz)] = Hazard(camera, zone, hz, float(g["prob"]), float(g["threshold"]), now)
            return self._transitions(now)

    def _active(self, now: float, hazard: str) -> List[Hazard]:
        return [h for h in self._hazards.values() if h.hazard == hazard and now - h.t <= self.ttl_s]

    def _evidence(self, ex: ExitInfo, now: float, hazard: str) -> Optional[Hazard]:
        hs = [h for h in self._active(now, hazard) if h.camera in ex.cameras or (ex.zone and h.zone == ex.zone)]
        return max(hs, key=lambda h: h.prob) if hs else None

    def _transitions(self, now: float) -> List[dict]:
        out = []
        for name, ex in self.exits.items():
            ev = self._evidence(ex, now, "fire_smoke")
            if ev is not None and name not in self._unsafe_since:
                self._unsafe_since[name] = now
                out.append({"type": "exit_blocked", "exit": name, "evidence": ev})
            elif ev is None and name in self._unsafe_since:
                del self._unsafe_since[name]
                out.append({"type": "exit_clear", "exit": name})
        return out

    # ------------------------------------------------------------ queries
    def exit_status(self, now: Optional[float] = None) -> Dict[str, dict]:
        now = time.time() if now is None else now
        with self._lock:
            out = {}
            for name, ex in self.exits.items():
                fire = self._evidence(ex, now, "fire_smoke")
                crowd = self._evidence(ex, now, "crowd")
                if fire is not None:
                    st = {"state": "unsafe", "reason": self._why(fire)}
                elif crowd is not None:
                    st = {"state": "congested", "reason": self._why(crowd)}
                else:
                    st = {"state": "clear" if ex.cameras or ex.zone else "not watched", "reason": ""}
                out[name] = {**st, "cameras": ex.cameras, "zone": ex.zone}
            for zn in sorted(set(self.camera_zones.values())):  # exits of watched zones that no camera sees: say so
                for e in (self.zones.get(zn) or {}).get("exits", []) or []:
                    if e not in out and self.exits:
                        out[e] = {"state": "not watched", "reason": "", "cameras": [], "zone": ""}
            return out

    @staticmethod
    def _why(h: Hazard) -> str:
        what = "smoke / fire" if h.hazard == "fire_smoke" else "abnormal crowd movement"
        return f"{what} seen by camera {h.camera} ({h.prob:.0%}, alert line {h.threshold:.0%})"

    def route_advice(self, zone: str, now: Optional[float] = None) -> Tuple[List[str], List[Tuple[str, str]]]:
        """(exits to use, in the zone map's order; [(exit to avoid, reason)]) for ``zone``."""
        status = self.exit_status(now)
        listed = list((self.zones.get(zone) or {}).get("exits", []) or [])
        use, avoid = [], []
        for e in listed:
            st = status.get(e, {"state": "not watched"})
            if st["state"] == "unsafe":
                avoid.append((e, st["reason"]))
            else:
                use.append(e)
        # congested exits go last: still usable, but the clear ones first
        use.sort(key=lambda e: status.get(e, {}).get("state") == "congested")
        return use, avoid

    def zone_hazard(self, place: str, now: Optional[float] = None) -> Optional[Hazard]:
        """Active fire/smoke at a place (an exit name or a zone name)."""
        now = time.time() if now is None else now
        with self._lock:
            if place in self.exits:
                return self._evidence(self.exits[place], now, "fire_smoke")
            hs = [h for h in self._active(now, "fire_smoke") if h.zone == place]
            return max(hs, key=lambda h: h.prob) if hs else None

    def crowd_toward_hazard(self, camera: str, flows: Dict[str, dict], now: Optional[float] = None) -> List[dict]:
        """``flows``: {place: {people, eta_median_s}} from one camera. Returns alerts to raise (rate-limited)."""
        now = time.time() if now is None else now
        out = []
        zone = self.camera_zones.get(camera, "")
        for place, f in flows.items():
            if f.get("people", 0) < self.min_people:
                continue
            eta = f.get("eta_median_s")
            if eta is not None and eta > self.max_eta_s:
                continue
            hz = self.zone_hazard(place, now)
            if hz is None:
                continue
            key = (camera, place)
            with self._lock:
                if now - self._last_crowd_alert.get(key, -1e18) < self.cooldown_s:
                    continue
                self._last_crowd_alert[key] = now
            use, _ = self.route_advice(zone, now)
            out.append({"camera": camera, "zone": zone, "place": place, "people": int(f["people"]),
                        "eta_s": eta, "evidence": hz, "alternatives": [e for e in use if e != place]})
        return out
