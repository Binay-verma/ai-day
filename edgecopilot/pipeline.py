"""The edge copilot: perception -> evidence fusion -> explanation -> action.

One :class:`Copilot` instance serves one camera (plus the sensors of its
zone).  ``process(frame, t)`` is the whole per-frame loop and returns a
:class:`FrameResult` with assessments, any *new* alerts (already explained,
verified and paired with recommendations) and per-stage timings.
"""
from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .copilot.events import Alert, EventLog
from .copilot.recommender import Recommender
from .io.sensors import SensorHub
from .perception.crowd import CrowdFeatureExtractor
from .perception.detector import Detection, DetectorOutput, OnnxYolo
from .perception.fall import FallFeatureExtractor
from .perception.smoke import SmokeCueExtractor
from .perception.tracker import Tracker, iou_matrix
from .privacy import anonymize, draw_skeleton
from .xai.calibration import ConformalGate
from .xai.counterfactual import explain_decision
from .xai.fusion import Assessment, HazardModel, Hysteresis
from .xai.nlg import explain, numbers_in, verify
from .xai.saliency import box_renormalize, eigencam, overlay, skeleton_saliency
from .xai.triage import triage

cv2.setNumThreads(1)  # avoid oversubscribing the edge CPU (ORT owns the cores)

HAZARD_COLORS = {"fall": (60, 60, 255), "fire_smoke": (0, 140, 255), "crowd": (255, 120, 40)}


def load_config(path: str) -> dict:
    """Raw config dict, resolving ``extends:`` overlays (validation: config_schema.load_config_dict)."""
    from .config_schema import load_raw
    return load_raw(path)


def _jpg_b64(img: np.ndarray, width: int = 480, quality: int = 70) -> str:
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


@dataclass
class FrameResult:
    t: float
    people: List[Detection]
    hazards: List[Detection]
    assessments: List[Assessment]
    new_alerts: List[Alert]
    timings: Dict[str, float]
    ran_detector: bool
    motion: float
    scene_change: bool = False
    privacy_people: List[Detection] = field(default_factory=list)  # every person detection, for anonymisation
    values: Dict[str, Dict[str, float]] = field(default_factory=dict)


class Copilot:
    def __init__(self, cfg: dict, root: str = ".", zone: str = "default",
                 sensors: Optional[SensorHub] = None, event_log: Optional[EventLog] = None,
                 pose_model: Optional[OnnxYolo] = None, fire_model: Optional[OnnxYolo] = None,
                 enable: Optional[Dict[str, bool]] = None):
        self.cfg = cfg
        self.root = root
        self.zone = zone
        rt = cfg.get("runtime", {})
        self.threads = rt.get("threads", 2)
        self.privacy = rt.get("privacy", "blur")
        self.motion_gate = rt.get("motion_gate", True)
        self.keepalive = rt.get("gate_keepalive", 8)
        self.fire_every = max(1, rt.get("fire_every", 2))
        self.fire_keepalive_s = rt.get("fire_keepalive_s", 1.0)
        self._since_fire = 10 ** 9
        self._last_fire_t = -1e9
        self._privacy_people: List[Detection] = []
        self.max_width = rt.get("max_width", 960)
        self.scene_cut = rt.get("scene_cut", 0.75)
        self.cooldown_s = rt.get("realert_cooldown_s", 30.0)
        self._closed: Dict[str, tuple] = {}  # key -> (alert, t_closed) for re-alert cooldown
        self._fall_box: Dict[str, np.ndarray] = {}  # last box of each open/closed fall incident (re-identification)
        self.exposure_jump = rt.get("exposure_jump", 0.45)
        self._unstable_until = -1.0
        self.scene_changes = 0
        self.enable = {"fall": True, "fire_smoke": True, "crowd": True}
        self.enable.update(enable or {})

        mc = cfg.get("models", {})
        self.pose = pose_model or self._load(mc.get("pose"), ["person"], pose=True)
        self.fire = fire_model or self._load(mc.get("fire_smoke"), mc.get("fire_smoke", {}).get("classes", ["fire", "smoke"]))
        self.models: Dict[str, HazardModel] = {hz: HazardModel.from_config(hz, c) for hz, c in cfg["hazards"].items()}
        self.hyst = Hysteresis(**cfg.get("hysteresis", {}))
        fb = cfg.get("feedback", {})
        self.fb_lr, self.fb_max = fb.get("lr", 0.5), fb.get("max_shift", 2.0)
        self.recommender = Recommender.from_config(cfg)
        # conformal false-alarm budgets (per hazard for this camera's zone)
        self.gates: Dict[str, ConformalGate] = {}
        self._calib_lock = threading.Lock()
        self.calib_path = os.path.join(root, "calibration", f"{zone.replace(' ', '_')}.json")
        self._load_calibration()
        self.sensors = sensors or SensorHub()
        self.events = event_log or EventLog()
        self.tracker = Tracker(high_thr=rt.get("track_high_conf", 0.35))
        self.fall_fx = FallFeatureExtractor()
        self.crowd_fx = CrowdFeatureExtractor()
        self.smoke_fx = SmokeCueExtractor()
        self.listeners: List[Callable[[Alert], None]] = []
        self.status_listeners: List[Callable[[Alert, str], None]] = []
        self.active: Dict[str, Alert] = {}
        self._prev_small: Optional[np.ndarray] = None
        self._since_heavy = 10 ** 9
        self._frame_idx = 0
        self._last_people: List[Detection] = []
        self._last_fire: Optional[DetectorOutput] = None
        self._last_frame: Optional[np.ndarray] = None
        self.last_result: Optional[FrameResult] = None

    # ------------------------------------------------------------------ setup
    def _load(self, mcfg: Optional[dict], classes, pose: bool = False) -> Optional[OnnxYolo]:
        if not mcfg:
            return None
        path = mcfg["path"] if os.path.isabs(mcfg["path"]) else os.path.join(self.root, mcfg["path"])
        if not os.path.exists(path):
            return None
        return OnnxYolo(path, classes, imgsz=mcfg.get("imgsz", 640), conf=mcfg.get("conf", 0.25),
                        iou=mcfg.get("iou", 0.45), pose=pose, threads=self.threads, sha256=mcfg.get("sha256"))

    def _load_calibration(self) -> None:
        import json
        import logging
        if os.path.exists(self.calib_path):
            try:
                with open(self.calib_path) as f:
                    data = json.load(f)
                for hz, g in data.items():
                    if hz in self.models:
                        scores = [float(x) for x in g["scores"] if np.isfinite(float(x))]
                        self.gates[hz] = ConformalGate(float(g["alpha"]), scores=scores)
            except (OSError, ValueError, KeyError, TypeError) as e:
                # a damaged calibration file must never stop the safety system: fall back to defaults, loudly
                logging.getLogger("edgecopilot.pipeline").error(
                    "calibration file %s unreadable (%s) - using configured thresholds", self.calib_path, e)
                self.gates = {}
        self._apply_gates()

    def _apply_gates(self) -> None:
        """Alert line for each hazard on this camera.

        Without calibration: the configured threshold.  With a calibrated
        conformal gate: the conformal quantile for the stated false-alarm budget
        (it may be lower than the default on a calm camera - that is how the
        indoor UMN scene went from 0/4 to 4/4 events detected - or higher on a busy
        one), but never below ``min_threshold`` so a degenerate calibration set
        cannot make a camera alert on noise, and never above ``max_threshold`` so
        a run of dismissals (or a stolen operator token) cannot silence a hazard.
        """
        for hz, m in self.models.items():
            hcfg = self.cfg["hazards"][hz]
            base = hcfg.get("threshold", 0.5)
            floor = hcfg.get("min_threshold", 0.2)
            ceil = max(hcfg.get("max_threshold", 0.9), base)
            g = self.gates.get(hz)
            tau = g.tau if g else None
            m.threshold = float(np.clip(1 / (1 + np.exp(-tau)), floor, ceil)) if tau is not None else base

    def _same_fall_incident(self, subject: str, t: float) -> Optional[str]:
        """Key of a recently closed fall incident at the same place as ``subject``'s track, if any."""
        try:
            tr = self.tracker.tracks.get(int(subject.split("#")[-1]))
        except ValueError:
            return None
        if tr is None:
            return None
        box = tr.last.box
        best, best_iou = None, 0.0
        for k, (al, tc) in self._closed.items():
            old = self._fall_box.get(k)
            if not k.startswith("fall:") or old is None or t - tc > self.cooldown_s or al.status not in ("resolved", "open"):
                continue
            iou = float(iou_matrix(old[None], box[None])[0, 0])
            if iou > best_iou:
                best, best_iou = k, iou
        return best if best_iou >= 0.3 else None

    def save_calibration(self) -> None:
        import json
        import tempfile
        os.makedirs(os.path.dirname(self.calib_path), exist_ok=True)
        with self._calib_lock:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.calib_path), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump({hz: g.to_dict() for hz, g in self.gates.items()}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.calib_path)  # atomic: readers see the old or the new file, never a torn one

    def reset(self) -> None:
        self.tracker.reset()
        self.fall_fx.reset()
        self.crowd_fx.reset()
        self.smoke_fx.reset()
        self.hyst.reset()
        self.active.clear()
        self._closed.clear()
        self._fall_box.clear()
        self._prev_small = None
        self._unstable_until = -1.0
        self._since_heavy = 10 ** 9
        self._frame_idx = 0
        self._last_people = []
        self._last_fire = None
        self._last_fire_t = -1e9
        self._since_fire = 10 ** 9

    def _reset_scene(self) -> None:
        self.tracker.reset()
        self.fall_fx.reset()
        self.crowd_fx.reset()
        self.smoke_fx.reset()
        self.hyst.reset()
        for al in self.active.values():
            if al.status == "open":
                al.status = "resolved"
        self.active.clear()
        self._closed.clear()
        self._fall_box.clear()
        self._last_people = []
        self._last_fire = None
        self._last_fire_t = -1e9
        self._since_fire = 10 ** 9

    @property
    def capacity(self) -> float:
        z = self.recommender.zones.get(self.zone)
        return float(z.capacity) if z else 50.0

    # ------------------------------------------------------------- main loop
    @staticmethod
    def normalise_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
        """Any camera frame -> 3-channel uint8 BGR, longest side <= max_side, shortest side >= 64."""
        if frame is None or frame.size == 0:
            raise ValueError("empty frame")
        if frame.dtype != np.uint8:
            f = frame.astype(np.float64)
            lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
            frame = np.nan_to_num((f - lo) * (255.0 / max(hi - lo, 1e-12))).clip(0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            frame = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        if min(h, w) < 64:
            frame = cv2.copyMakeBorder(frame, 0, max(0, 64 - h), 0, max(0, 64 - w), cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return np.ascontiguousarray(frame)

    def process(self, frame: np.ndarray, t: float, render: bool = False, wall_t: Optional[float] = None) -> FrameResult:
        """Process one frame. ``t`` must be monotonic (kinematics); ``wall_t`` (default ``t``) is the clock
        that sensor readings are stamped with."""
        timings: Dict[str, float] = {}
        t_all = time.perf_counter()
        frame = self.normalise_frame(frame, self.max_width)
        wall_t = t if wall_t is None else wall_t
        self._frame_idx += 1

        # 1) motion gate (cascade): cheap frame difference decides if the
        #    heavy detectors need to run on this frame
        t0 = time.perf_counter()
        small = cv2.cvtColor(cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        motion = 1.0 if self._prev_small is None else float((cv2.absdiff(small, self._prev_small) > 12).mean())
        self._prev_small = small
        scene_change = motion > self.scene_cut and self._frame_idx > 2
        if scene_change:
            # camera moved / feed switched / tampering: old baselines and tracks are meaningless
            self._reset_scene()
            self.scene_changes += 1
        any_active = bool(self.active)
        run_heavy = (not self.motion_gate) or motion > 0.004 or self._since_heavy >= self.keepalive or any_active
        timings["gate"] = (time.perf_counter() - t0) * 1e3

        # 2) people + pose
        t0 = time.perf_counter()
        need_pose = self.enable["fall"] or self.enable["crowd"] or self.privacy != "raw"
        if run_heavy and self.pose is not None and need_pose:
            out = self.pose(frame)
            people = [d for d in out.detections]
            self._last_people = people
            self._since_heavy = 0
        else:
            people = [Detection(d.box.copy(), d.score, d.cls, d.label, None if d.keypoints is None else d.keypoints.copy())
                      for d in self._last_people]
            self._since_heavy += 1
        timings["pose"] = (time.perf_counter() - t0) * 1e3
        # privacy: EVERY person detection (tracked or not, any confidence) is anonymised
        self._privacy_people = list(people)
        t0 = time.perf_counter()
        tracks = self.tracker.update(people, t)
        # analytics: low-confidence detections only count if they continued an existing track (ByteTrack)
        people = [d for d in people if d.track_id >= 0]
        timings["track"] = (time.perf_counter() - t0) * 1e3

        assessments: List[Assessment] = []
        values_all: Dict[str, Dict[str, float]] = {}
        sens_vals, sens_rel = self.sensors.cues(self.zone, wall_t)

        # 3) fall cues per tracked person
        t0 = time.perf_counter()
        if self.enable["fall"]:
            live = set()
            for tr in tracks:
                live.add(tr.track_id)
                if tr.hits < 2:
                    continue
                vals, rel = self.fall_fx(tr, t)
                a = self.models["fall"].assess(f"person#{tr.track_id}", t, vals, rel, zone_key=self.zone)
                assessments.append(self.hyst.update(f"fall:{tr.track_id}", a))
                values_all[a.subject] = vals
            self.fall_fx.prune(set(self.tracker.tracks))
            self.hyst.prune("fall:", {f"fall:{i}" for i in self.tracker.tracks}, t)
        timings["fall"] = (time.perf_counter() - t0) * 1e3

        # 4) crowd cues for the zone
        t0 = time.perf_counter()
        crowd_vals: Dict[str, float] = {}
        if self.enable["crowd"] or self.enable["fire_smoke"]:
            recent_n = max([c for ts, c in self.crowd_fx.count_hist if t - ts <= 3.0] + [len(people)])
            if recent_n >= 1 or self.pose is None:
                heights = [d.wh[1] for d in people if d.wh[1] > 10]
                ph = float(np.median(heights)) if heights else None
                crowd_vals = self.crowd_fx(frame, t, len(people), self.capacity, ph,
                                           alert_active=any(k.startswith("crowd") for k in self.active))
            else:  # empty scene: crowd cues are meaningless, skip the optical flow (largest CPU saving)
                crowd_vals = self.crowd_fx.idle(t, len(people))
        # a moving camera invalidates every "compared with this scene's normal" cue
        ego = crowd_vals.get("ego_motion", 0.0)
        cam_rel = float(np.clip(1.0 - (ego - 0.02) / 0.08, 0.2, 1.0))
        # a sudden global change below the scene-cut level (auto-exposure jump, lights
        # switched, camera bumped) also makes flow / background cues meaningless for ~1 s
        if motion > self.exposure_jump:
            self._unstable_until = t + 1.0
        if t < self._unstable_until:
            cam_rel = 0.2
        if self.enable["crowd"]:
            cv_ = dict(crowd_vals)
            if "sound_db" in sens_vals:
                cv_["sound_db"] = sens_vals["sound_db"]
            rel_c = dict(sens_rel)
            # crowd motion cues describe *people*: flames, trees or a moving robot are not a crowd
            recent_people = max([c for _, c in self.crowd_fx.count_hist] + [0])
            people_rel = 1.0 if recent_people >= 2 else 0.0
            rel_c.update({k: cam_rel * people_rel for k in ("surge", "divergence", "coherence", "exodus")})
            a = self.models["crowd"].assess(self.zone, t, cv_, rel_c, zone_key=self.zone)
            assessments.append(self.hyst.update("crowd", a))
            values_all["crowd"] = cv_
        timings["crowd"] = (time.perf_counter() - t0) * 1e3

        # 5) smoke / fire: detector every `fire_every` frames + cheap image cues every frame
        t0 = time.perf_counter()
        hazards: List[Detection] = []
        if self.enable["fire_smoke"]:
            fire_active = any(k.startswith("fire") for k in self.active)
            self._since_fire += 1
            due = self._since_fire >= self.fire_every and (run_heavy or fire_active)
            stale = t - self._last_fire_t >= self.fire_keepalive_s  # slow smoke barely moves: never rely on the gate alone
            if self.fire is not None and (due or stale or self._last_fire is None):
                self._last_fire = self.fire(frame)
                self._since_fire = 0
                self._last_fire_t = t
            if self._last_fire is not None:
                hazards = self._last_fire.detections
            fv = self.smoke_fx(frame, t, alert_active=fire_active)
            if self._last_fire is not None and t - self._last_fire_t > 3 * self.fire_keepalive_s:
                hazards = []  # never explain with a detection older than a few seconds
            fv["smoke_conf"] = max([d.score for d in hazards if d.label == "smoke"], default=0.0)
            fv["fire_conf"] = max([d.score for d in hazards if d.label == "fire"], default=0.0)
            if crowd_vals and "coherence" in crowd_vals and self.crowd_fx.count_hist and \
                    max(c for _, c in self.crowd_fx.count_hist) >= 2:
                # people running (surge) AND either all one way (coherence) or leaving the view (exodus)
                fv["rapid_exit"] = float(np.clip((crowd_vals["surge"] - 1.0) / 3.0, 0, 1)
                                         * max(crowd_vals["coherence"], crowd_vals["exodus"]))
            for k in ("smoke_ppm", "temp_rise"):
                if k in sens_vals:
                    fv[k] = sens_vals[k]
            rel = dict(sens_rel)
            rel.update({k: cam_rel for k in ("smoke_density", "smoke_growth", "fire_flicker", "rapid_exit")})
            if self.fire is None:
                rel.update({"smoke_conf": 0.0, "fire_conf": 0.0})
            a = self.models["fire_smoke"].assess(self.zone, t, fv, rel, zone_key=self.zone)
            assessments.append(self.hyst.update("fire_smoke", a))
            values_all["fire_smoke"] = fv
        timings["fire_smoke"] = (time.perf_counter() - t0) * 1e3

        # 6) alerts: explain + verify + recommend on onset only
        t0 = time.perf_counter()
        new_alerts: List[Alert] = []
        for a in assessments:
            key = f"{a.hazard}:{a.subject}"
            recent = self._closed.get(key)
            if a.onset and recent is None and a.hazard == "fall":
                # the tracker may have lost and re-acquired the same fallen person under a new id:
                # same place, within the cooldown -> same incident
                ckey = self._same_fall_incident(a.subject, t)
                if ckey is not None:
                    recent = self._closed.pop(ckey)
                    self._fall_box.pop(ckey, None)
            if a.onset and recent is not None and t - recent[1] <= self.cooldown_s and recent[0].status in ("resolved", "open"):
                # same incident flaring up again: re-open it instead of paging the operator twice
                al = recent[0]
                al.status = "open"
                al.peak_prob = max(al.peak_prob, a.prob)
                al.peak_ema = max(al.peak_ema, a.ema_logit if a.ema_logit is not None else a.logit)
                self._notify_status(al, "reopened")
                self.active[key] = al
            elif a.onset:
                alert = self._make_alert(a, frame, people, hazards, timings)
                self.active[key] = alert
                self.events.add(alert)
                new_alerts.append(alert)
                for cb in self.listeners:
                    cb(alert)
            elif a.active and key in self.active:
                self.active[key].peak_prob = max(self.active[key].peak_prob, a.prob)
                self.active[key].peak_ema = max(self.active[key].peak_ema, a.ema_logit if a.ema_logit is not None else a.logit)
            elif not a.active and key in self.active:
                al = self.active.pop(key)
                if al.status == "open":
                    al.status = "resolved"
                self._closed[key] = (al, t)
                self._notify_status(al, "cleared")
        # tracks that disappeared close their fall alerts (and start the re-alert cooldown)
        for key in [k for k in self.active if k.startswith("fall:")]:
            tid = int(key.split("#")[-1])
            live = self.tracker.tracks.get(tid)
            if live is not None:
                self._fall_box[key] = live.last.box.copy()
            else:
                al = self.active.pop(key)
                if al.status == "open":
                    al.status = "resolved"
                self._closed[key] = (al, t)
                self._notify_status(al, "cleared")
        # bounded memory: forget closed incidents once their re-open window has passed
        for k in [k for k, (_, tc) in self._closed.items() if t - tc > self.cooldown_s]:
            del self._closed[k]
        for k in [k for k in self._fall_box if k not in self._closed and k not in self.active]:
            del self._fall_box[k]
        timings["explain"] = (time.perf_counter() - t0) * 1e3
        timings["total"] = (time.perf_counter() - t_all) * 1e3
        self._last_frame = frame
        res = FrameResult(t, people, hazards, assessments, new_alerts, timings, run_heavy, motion, scene_change,
                          values=values_all, privacy_people=self._privacy_people)
        self.last_result = res
        return res

    # ------------------------------------------------------------ explaining
    def _evidence_map(self, a: Assessment, frame: np.ndarray, people: List[Detection], hazards: List[Detection]) -> np.ndarray:
        if a.hazard == "fall":
            tid = int(a.subject.split("#")[-1])
            kps = [d.keypoints for d in people if d.track_id == tid and d.keypoints is not None]
            sal = skeleton_saliency(frame.shape, kps)
            boxes = [d.box for d in people if d.track_id == tid]
            if boxes:
                sal = np.maximum(sal, 0.35 * box_renormalize(np.ones(frame.shape[:2], np.float32), boxes))
            return sal
        if a.hazard == "fire_smoke":
            fo = self._last_fire
            if fo is not None and fo.feature is not None:
                cam = eigencam(fo.feature, fo.letterbox, frame.shape)
                return box_renormalize(cam, [d.box for d in hazards]) if hazards else cam
            dm = self.smoke_fx.cell_delta
            if dm is not None:
                return cv2.resize(np.clip(dm / (dm.max() + 1e-9), 0, 1).astype(np.float32),
                                  (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        if a.hazard == "crowd" and self.crowd_fx.heatmap is not None:
            return cv2.resize(self.crowd_fx.heatmap, (frame.shape[1], frame.shape[0]))
        return np.zeros(frame.shape[:2], np.float32)

    def _make_alert(self, a: Assessment, frame, people, hazards, timings) -> Alert:
        t0 = time.perf_counter()
        model = self.models[a.hazard]
        rec = self.recommender.recommend(a, self.zone)
        rec_texts = [x["text"] for x in rec["actions"][:3]]
        tri = triage(a, **self.cfg.get("triage", {}))
        exp = explain(a, model, self.zone, rec_texts, triage=tri)
        ver = verify(exp.text, a, model, self.zone, extra_numbers=numbers_in(rec_texts), recommendations=rec_texts)
        dec = explain_decision(a, model)
        sal = self._evidence_map(a, frame, people, hazards)
        safe = anonymize(frame, self._privacy_people, self.privacy, pose_available=self.pose is not None)
        ev = overlay(safe, sal)
        for d in hazards:
            x1, y1, x2, y2 = d.box.astype(int)
            cv2.rectangle(ev, (x1, y1), (x2, y2), (0, 140, 255), 2)
        explain_ms = (time.perf_counter() - t0) * 1e3
        return Alert(
            hazard=a.hazard, subject=a.subject, zone=self.zone, t_video=a.t, prob=a.prob,
            severity=rec["severity"], explanation=exp.to_dict(), decision=dec,
            verification=ver.to_dict(), recommendations=rec["actions"],
            cues=[c.to_dict() for c in sorted(a.cues, key=lambda c: -c.contribution)],
            snapshot_jpg_b64=_jpg_b64(self.render(frame, people, hazards, [a])),
            evidence_jpg_b64=_jpg_b64(ev), processing_ms=sum(v for k, v in timings.items() if k != "total"),
            explain_ms=explain_ms, peak_prob=a.prob, triage=tri,
            peak_ema=a.ema_logit if a.ema_logit is not None else a.logit,
        )

    # ---------------------------------------------------------- feedback loop
    def _notify_status(self, al: Alert, change: str) -> None:
        for cb in self.status_listeners:
            try:
                cb(al, change)
            except Exception:  # noqa: BLE001 - persistence problems must not break the frame loop
                import logging
                logging.getLogger("edgecopilot.pipeline").exception("status listener failed")

    def feedback(self, alert_id: str, confirmed: bool, note: str = "") -> Optional[dict]:
        prev = self.events.alerts.get(alert_id)
        if prev is None or prev.status in ("confirmed", "dismissed"):
            return None  # each alert can be labelled once: repeated dismissals must not desensitise a camera
        al = self.events.update_status(alert_id, "confirmed" if confirmed else "dismissed", note)
        if al is None:
            return None
        out: dict = {"id": alert_id, "hazard": al.hazard, "zone": al.zone}
        if al.hazard in self.gates:
            # calibrated camera: a dismissal is one more labelled *normal window*, in the same units as the
            # calibration data (peak smoothed logit); the intercept is NOT also shifted (no double counting)
            if not confirmed:
                self.gates[al.hazard].add_normal(float(al.peak_ema))
                self._apply_gates()
                self.save_calibration()
            out["threshold"] = self.models[al.hazard].threshold
        else:
            out["bias_offset"] = self.models[al.hazard].feedback(al.zone, int(confirmed), al.prob, self.fb_lr, self.fb_max)
        return out

    # -------------------------------------------------------------- rendering
    def render(self, frame: np.ndarray, people: List[Detection], hazards: List[Detection],
               assessments: List[Assessment]) -> np.ndarray:
        img = anonymize(frame, self._privacy_people, self.privacy, pose_available=self.pose is not None)
        active_people = {a.subject for a in assessments if a.hazard == "fall" and (a.active or a.prob > 0.5)}
        for d in people:
            col = (60, 60, 255) if f"person#{d.track_id}" in active_people else (90, 200, 90)
            x1, y1, x2, y2 = d.box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            if d.keypoints is not None and self.privacy != "skeleton":
                draw_skeleton(img, d.keypoints)
            cv2.putText(img, f"#{d.track_id}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        for d in hazards:
            x1, y1, x2, y2 = d.box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(img, f"{d.label} {d.score:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
        y = 22
        for a in assessments:
            if a.active:
                col = HAZARD_COLORS.get(a.hazard, (255, 255, 255))
                txt = f"{a.hazard.upper()} {a.subject} p={a.prob:.2f}"
                cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
                y += 22
        return img
