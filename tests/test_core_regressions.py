"""Regression tests for the independent core-code audit (docs/research/audit_core.md, findings C1-C20)."""
import json
import os

import cv2
import numpy as np
import pytest

from edgecopilot.copilot.events import Alert
from edgecopilot.copilot.recommender import Recommender, safe_eval
from edgecopilot.io.sensors import SensorHub
from edgecopilot.perception.crowd import CrowdFeatureExtractor
from edgecopilot.perception.detector import Detection
from edgecopilot.perception.smoke import SmokeCueExtractor
from edgecopilot.perception.tracker import Tracker
from edgecopilot.pipeline import Copilot, load_config
from edgecopilot.xai.fusion import HazardModel, Hysteresis
from edgecopilot.xai.nlg import explain, verify

CFG = load_config("configs/campus.yaml")
HAVE_POSE = os.path.exists("models/yolo11n-pose.onnx")


def no_models(cfg=None):
    import copy
    c = copy.deepcopy(cfg or CFG)
    for k in c["models"]:
        c["models"][k]["path"] = "models/__missing__.onnx"
    return c


def face_frame():
    import ultralytics
    return cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))


# C1 - fire-only camera still anonymises people
@pytest.mark.skipif(not HAVE_POSE, reason="pose model not downloaded")
def test_c1_privacy_on_fire_only_camera(tmp_path):
    cop = Copilot(CFG, root=".", zone="Lab Block A", enable={"fall": False, "crowd": False, "fire_smoke": True})
    img = face_frame()
    res = cop.process(img, 0.0)
    out = cop.render(img, res.people, res.hazards, res.assessments)
    assert res.privacy_people, "person detector must run when privacy filtering is on"
    from edgecopilot.privacy import head_regions
    x1, y1, x2, y2 = head_regions(res.privacy_people)[0]
    x1, y1 = max(0, x1), max(0, y1)
    assert np.abs(out[y1:y2, x1:x2].astype(int) - img[y1:y2, x1:x2].astype(int)).mean() > 5


def test_c1_fail_safe_without_person_model():
    from edgecopilot.privacy import anonymize
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    out = anonymize(img, [], "blur", pose_available=False)
    assert cv2.Laplacian(out, cv2.CV_64F).var() < 0.1 * cv2.Laplacian(img, cv2.CV_64F).var()


# C2 - untracked low-confidence people are anonymised too
def test_c2_low_confidence_people_are_in_privacy_list():
    cop = Copilot(no_models(), root=".", zone="Corridor B")
    low = Detection(np.array([10, 10, 60, 150.0]), 0.2, 0, "person", np.zeros((17, 3)))
    cop._last_people = [low]
    cop.motion_gate = True
    cop._since_heavy = 0
    frame = np.zeros((240, 320, 3), np.uint8)
    cop.process(frame, 0.0)
    res = cop.process(frame, 0.1)
    assert res.people == [] and len(res.privacy_people) == 1


# C3 - "a or b" with an unmeasured cue still fires on the measured one
def test_c3_missing_cue_does_not_kill_or_condition():
    assert safe_eval("prob > 0.85 or smoke_ppm > 600", {"prob": 0.95})
    assert not safe_eval("prob > 0.85 and smoke_ppm > 600", {"prob": 0.95})
    assert not safe_eval("-smoke_ppm < 5", {})
    m = HazardModel.from_config("fire_smoke", CFG["hazards"]["fire_smoke"])
    a = m.assess("Lab Block A", 0, {"smoke_conf": 0.95, "fire_conf": 0.95, "smoke_density": 0.4})
    acts = " ".join(x["text"] for x in Recommender.from_config(CFG).recommend(a, "Lab Block A")["actions"])
    assert "Trigger evacuation" in acts and a.prob > 0.85


# C17 - offline sensors do not trigger playbook steps
def test_c17_offline_sensor_ignored_by_playbook():
    m = HazardModel.from_config("fire_smoke", CFG["hazards"]["fire_smoke"])
    a = m.assess("Lab Block A", 0, {"smoke_conf": 0.6, "smoke_ppm": 900}, {"smoke_ppm": 0.0})
    acts = " ".join(x["text"] for x in Recommender.from_config(CFG).recommend(a, "Lab Block A")["actions"])
    assert "Trigger evacuation" not in acts


# C4 - slowly building smoke is not absorbed into the background
@pytest.mark.parametrize("build_s", [1.0, 10.0, 30.0])
def test_c4_slow_smoke_still_detected(build_s):
    fx = SmokeCueExtractor()
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    fps, peak = 10.0, 0.0
    for i in range(int((5 + build_s + 5) * fps)):
        t = i / fps
        a = float(np.clip((t - 5) / build_s, 0, 1)) * 0.6
        f = base.copy()
        f[:120, :160] = (f[:120, :160] * (1 - a) + 200 * a).astype(np.uint8)
        peak = max(peak, fx(f, t)["smoke_density"])
    assert peak > 0.2, f"smoke built over {build_s}s reached only {peak:.3f}"


# C5 - fire detector keeps running in a still scene
def test_c5_fire_detector_runs_on_static_scene():
    calls = []

    class FakeFire:
        def __call__(self, frame):
            from edgecopilot.perception.detector import DetectorOutput
            calls.append(1)
            return DetectorOutput([], None, (1.0, 0, 0, 640, 640), 1.0)
    cfg = no_models()
    cfg["runtime"]["fire_every"] = 3
    cop = Copilot(cfg, root=".", zone="Lab Block A", fire_model=FakeFire(),
                  enable={"fall": False, "crowd": False, "fire_smoke": True})
    frame = np.full((240, 320, 3), 90, np.uint8)
    for i in range(300):  # 30 s of a perfectly still image at 10 fps
        cop.process(frame, i / 10)
    assert len(calls) >= 25  # at least ~1 Hz keep-alive


# C6 - an alert is never explained as below its own threshold
def test_c6_onset_frame_is_above_threshold():
    m = HazardModel.from_config("fall", CFG["hazards"]["fall"])
    h = Hysteresis(alpha=0.3, on_frames=2, off_frames=5)
    hi = {"torso_angle": 85, "descent_speed": 2.0, "height_drop": 0.7, "aspect_ratio": 2.5, "ground_time": 3, "tilt_rate": 200}
    lo = {"torso_angle": 5, "descent_speed": 0.0, "height_drop": 0.0, "aspect_ratio": 0.4, "ground_time": 0, "tilt_rate": 0}
    seq = [hi] * 6 + [lo] + [hi] * 3
    for vals in seq:
        a = h.update("k", m.assess("person#1", 0, vals))
        if a.onset:
            assert a.prob >= a.threshold
            e = explain(a, m, "Corridor B")
            assert "would not have fired" in e.text or "combination" in e.text


# C8 - isolated spikes far apart never add up to an alert
def test_c8_spikes_do_not_accumulate():
    m = HazardModel.from_config("fall", CFG["hazards"]["fall"])
    h = Hysteresis(alpha=1.0, on_frames=2, off_frames=5, off_margin=0.15)
    spike = {"torso_angle": 85, "descent_speed": 2.0, "height_drop": 0.7, "aspect_ratio": 2.5, "ground_time": 3, "tilt_rate": 200}
    mid = {"torso_angle": 84, "descent_speed": 1.3, "height_drop": 0.55, "aspect_ratio": 1.6, "ground_time": 0, "tilt_rate": 90}
    assert 0.45 <= m.assess("p", 0, mid).prob < 0.6  # dead-band frame: below the line, above the off-margin
    onset = False
    for i in range(400):
        a = h.update("k", m.assess("p", 0, spike if i in (0, 200) else mid))
        onset |= a.onset
    assert not onset


# C7 - verifier false accepts found by the audit
@pytest.fixture()
def fall_alert():
    m = HazardModel.from_config("fall", CFG["hazards"]["fall"])
    vals = {"torso_angle": 82, "descent_speed": 1.8, "height_drop": 0.62, "aspect_ratio": 1.9, "ground_time": 0.0, "tilt_rate": 150}
    return m, m.assess("person#3", 0, vals)


@pytest.mark.parametrize("bad", [
    "Fall detected. The person has been lying motionless for 82 s.",      # value of another cue
    "Fall detected. The room is at 3 percent occupancy.",                  # track id read as a measurement
    "Fall detected because the person stayed motionless on the floor.",    # ground_time lowered the risk
    "Crowd stampede and panic detected near person#3.",                    # wrong hazard words
])
def test_c7_verifier_rejects(fall_alert, bad):
    m, a = fall_alert
    assert not verify(bad, a, m, "Corridor B").ok, bad


def test_c7_fell_on_crowd_alert_rejected():
    m = HazardModel.from_config("crowd", CFG["hazards"]["crowd"])
    a = m.assess("Auditorium", 0, {"surge": 6.0, "exodus": 0.8, "divergence": 1.0})
    assert not verify("A person fell in the Auditorium.", a, m, "Auditorium").ok
    e = explain(a, m, "Auditorium", ["Check for a triggering incident (fire/fall) near Auditorium"])
    assert verify(e.text, a, m, "Auditorium").ok  # playbook text may name other hazards


def test_c7_location_digits_and_paraphrase_ok(fall_alert):
    m, a = fall_alert
    e = explain(a, m, "Room B-104", ["Bring the AED outside Room B-104"])
    assert verify(e.text, a, m, "Room B-104", extra_numbers=[104]).ok


def test_c7_nothing_hides_after_recommended(fall_alert):
    """Claims appended after the playbook section are still checked as narrative."""
    m, a = fall_alert
    recs = ["Call security on ext. 2311", "Bring the AED"]
    e = explain(a, m, "Corridor B", recs)
    assert verify(e.text, a, m, "Corridor B", recommendations=recs).ok
    for tail in [" Smoke is spreading fast.", " The person has been lying still for 82 s.",
                 " A stampede broke out."]:
        assert not verify(e.text + tail, a, m, "Corridor B", recommendations=recs).ok, tail
        assert not verify(e.text + tail, a, m, "Corridor B", extra_numbers=[2311]).ok, tail
    # playbook text moved to the front is still recognised as playbook text
    import re as _re
    mm = _re.search(r"Recommended:.*$", e.text)
    moved = mm.group(0) + " " + e.text[:mm.start()]
    assert verify(moved, a, m, "Corridor B", recommendations=recs).ok


# C12 - any frame shape/format is accepted
@pytest.mark.parametrize("frame", [
    np.zeros((240, 320), np.uint8), np.zeros((240, 320, 4), np.uint8), np.zeros((240, 320, 1), np.uint8),
    np.zeros((20, 2000, 3), np.uint8), np.zeros((3000, 40, 3), np.uint8), np.zeros((10, 10, 3), np.uint8),
    np.zeros((240, 320, 3), np.uint16),
])
def test_c12_frame_formats(frame):
    cop = Copilot(no_models(), root=".", zone="Corridor B")
    for i in range(3):
        cop.process(frame, i / 10)


# C13 - first optical-flow pair of a file starting at t=0 is not inflated
def test_c13_crowd_first_frame_dt():
    fx = CrowdFeatureExtractor()
    a = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(a, (100, 100), 10, (255, 255, 255), -1)
    b = np.zeros_like(a)
    cv2.circle(b, (104, 100), 10, (255, 255, 255), -1)
    fx(a, 0.0, 2, person_height_px=50)
    v = fx(b, 0.04, 2, person_height_px=50)
    assert v["speed"] < 20  # ~4 px in 0.04 s over a 50 px person ~= 2 heights/s, not 66x that


# C14 - per-person state is bounded on long runs
def test_c14_hysteresis_state_pruned():
    cop = Copilot(no_models(), root=".", zone="Corridor B")
    for i in range(500):
        cop.hyst.state[f"fall:{i}"] = {"ema": -5, "above": 0, "below": 0, "active": False, "t": 0.0}
    cop.hyst.prune("fall:", set(), 100.0)
    assert not [k for k in cop.hyst.state if k.startswith("fall:")]


# C16 - dismissals add calibration samples in calibration units, without double counting
def test_c16_dismissal_uses_peak_ema_and_no_bias_shift(tmp_path):
    from edgecopilot.xai.calibration import ConformalGate
    cop = Copilot(no_models(), root=str(tmp_path), zone="Lab Block A")
    cop.gates["fire_smoke"] = ConformalGate(0.1, scores=[-3.0] * 20)
    al = Alert("fire_smoke", "Lab Block A", "Lab Block A", 1.0, 0.9, "HIGH", {}, {}, {}, [], [], peak_ema=1.25)
    cop.events.add(al)
    cop.feedback(al.id, confirmed=False)
    assert cop.gates["fire_smoke"].scores[-1] == pytest.approx(1.25)
    assert cop.models["fire_smoke"].bias_offset == {}


# C18 - clearing is persisted (latched alerts stay open until acknowledged)
def test_c18_cleared_persisted(tmp_path):
    from edgecopilot.config_schema import load_config_dict
    from edgecopilot.service import EdgeService
    from edgecopilot.storage import Store
    cfg = no_models(load_config_dict("configs/campus.yaml"))
    store = Store(str(tmp_path / "d.db"), str(tmp_path / "m"))
    svc = EdgeService(cfg, root=str(tmp_path), store=store, start_mqtt=False)
    al = Alert("fall", "person#1", "Corridor B", 1.0, 0.9, "HIGH", {}, {}, {}, [], [])
    store.save_alert({"id": al.id, "hazard": "fall", "status": "open", "wall_time": 1.0})
    svc._on_status(al, "cleared")
    row = store.get_alert(al.id)
    assert row["cleared_at"] is not None and row["status"] == "open"


# C19 - key-points never leave the image
@pytest.mark.skipif(not HAVE_POSE, reason="pose model not downloaded")
def test_c19_keypoints_clipped():
    from edgecopilot.perception.detector import OnnxYolo
    img = face_frame()
    m = OnnxYolo("models/yolo11n-pose.onnx", ["person"], pose=True, conf=0.3)
    for d in m(img).detections:
        assert d.keypoints[:, 0].min() >= 0 and d.keypoints[:, 0].max() <= img.shape[1] - 1
        assert d.keypoints[:, 1].min() >= 0 and d.keypoints[:, 1].max() <= img.shape[0] - 1


# C20 - track lifetime is in seconds (fps independent)
def test_c20_track_age_in_seconds():
    for fps in (5, 30):
        tr = Tracker(max_misses=15, max_age_s=1.5)
        d = Detection(np.array([0, 0, 50, 100.0]), 0.9, 0, "person")
        tr.update([d], 0.0)
        n_frames = int(1.0 * fps)  # 1 s of misses: the track must survive at any fps
        for i in range(1, n_frames + 1):
            tr.update([], i / fps)
        assert tr.tracks, f"track died after 1 s at {fps} fps"


def test_c9_standalone_hub_has_no_fallback():
    hub = SensorHub()
    hub.push({"zone": "default", "t": 1.0, "smoke_ppm": 900})
    assert hub.cues("Corridor B", 1.0) == ({}, {})


def test_calibration_json_written_atomically(tmp_path):
    from edgecopilot.xai.calibration import ConformalGate
    cop = Copilot(no_models(), root=str(tmp_path), zone="Lab Block A")
    cop.gates["fall"] = ConformalGate(0.1, scores=[0.0] * 30)
    cop.save_calibration()
    assert json.load(open(cop.calib_path))["fall"]["n"] == 30
