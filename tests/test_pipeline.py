import os

import cv2
import numpy as np
import pytest

from edgecopilot.copilot.events import EventLog
from edgecopilot.io.sensors import SensorHub, synthetic_fire_trace
from edgecopilot.perception.crowd import CrowdFeatureExtractor
from edgecopilot.perception.smoke import SmokeCueExtractor
from edgecopilot.pipeline import Copilot, load_config
from edgecopilot.privacy import anonymize
from edgecopilot.perception.detector import Detection
from edgecopilot.xai.fusion import HazardModel
from edgecopilot.xai.nlg import explain, verify

CFG = load_config("configs/campus.yaml")


def test_template_explanations_grounded_for_every_hazard():
    """Property test: whatever the cue values, the template text verifies 100 %."""
    rng = np.random.default_rng(0)
    for hz, hcfg in CFG["hazards"].items():
        m = HazardModel.from_config(hz, hcfg)
        for _ in range(300):
            vals = {k: float(rng.uniform(s.lo - 0.2 * abs(s.hi - s.lo), s.hi)) for k, s in m.specs.items()}
            a = m.assess("person#2" if hz == "fall" else "Lab Block A", 0.0, vals)
            if a.logit < a.logit_threshold:
                continue
            e = explain(a, m, "Lab Block A", ["Call 112"])
            v = verify(e.text, a, m, "Lab Block A", extra_numbers=[112])
            assert v.ok, (hz, e.text, v.to_dict())


def test_smoke_cues_respond_to_haze():
    fx = SmokeCueExtractor()
    rng = np.random.default_rng(0)
    base = (rng.uniform(0, 255, (240, 320, 3))).astype(np.uint8)
    for i in range(30):
        fx(base, i / 15)
    hazy = base.copy()
    hazy[:120, :160] = (0.35 * hazy[:120, :160] + 0.65 * 200).astype(np.uint8)  # grey veil top-left
    v = fx(hazy, 2.1)
    assert v["smoke_density"] > 0.2 and v["contrast_loss"] > 0.2


def test_crowd_surge_detected():
    fx = CrowdFeatureExtractor()
    img = np.zeros((240, 320, 3), np.uint8)
    rng = np.random.default_rng(0)
    blobs = rng.uniform(20, 300, (8, 2))
    vals = {}
    for i in range(120):
        speed = 1.0 if i < 90 else 6.0  # px/frame walking, then running
        if i >= 90:
            blobs += rng.normal(0, 1, blobs.shape) * 0 + np.sign(blobs - 160) * speed
        else:
            blobs[:, 0] += speed
        blobs %= 320
        frame = img.copy()
        for x, y in blobs:
            cv2.circle(frame, (int(x), int(y) % 240), 8, (255, 255, 255), -1)
        vals = fx(frame, i / 15, n_people=8, capacity=40, person_height_px=30)
    assert vals["surge"] > 2.5


def test_privacy_blur_changes_head_only():
    img = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    kp = np.zeros((17, 3))
    kp[0] = [100, 40, 0.9]
    kp[1] = [95, 35, 0.9]
    kp[2] = [105, 35, 0.9]
    d = Detection(np.array([70, 20, 130, 190.0]), 0.9, 0, "person", kp)
    out = anonymize(img, [d], "blur")
    assert not np.array_equal(out[30:50, 90:110], img[30:50, 90:110])
    assert np.array_equal(out[150:190, 0:60], img[150:190, 0:60])


def test_sensor_staleness():
    hub = SensorHub(stale_s=5)
    for r in synthetic_fire_trace("Lab", 0, 30, fire_at=10):
        hub.push(r)
    vals, rel = hub.cues("Lab", 30.5)
    assert vals["smoke_ppm"] > 600 and rel["smoke_ppm"] == 1.0 and vals["temp_rise"] > 0
    vals, rel = hub.cues("Lab", 60.0)
    assert rel["smoke_ppm"] == 0.0


def test_pipeline_runs_without_models(tmp_path):
    """The copilot degrades gracefully (cue-only) when model files are missing."""
    cfg = load_config("configs/campus.yaml")
    for k in cfg["models"]:
        cfg["models"][k]["path"] = "models/__missing__.onnx"
    hub = SensorHub()
    for r in synthetic_fire_trace("default", 0, 20, fire_at=3):
        hub.push(r)
    cop = Copilot(cfg, root=str(tmp_path), zone="Lab Block A", sensors=hub, event_log=EventLog(str(tmp_path / "ev.jsonl")))
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    alerts = []
    for i in range(200):
        f = frame.copy()
        if i > 40:  # flickering flame-coloured patch + growing grey haze
            f[100:160, 300:340] = (20, 120 + (i % 3) * 30, 250)
            a = min(0.8, (i - 40) / 60)
            f[:180] = (f[:180] * (1 - a) + 190 * a).astype(np.uint8)
        res = cop.process(f, i / 10)
        alerts += res.new_alerts
        assert res.timings["total"] > 0
    assert any(al.hazard == "fire_smoke" for al in alerts)
    al = next(al for al in alerts if al.hazard == "fire_smoke")
    assert al.verification["groundedness"] == 1.0
    assert al.evidence_jpg_b64 and al.recommendations
    fb = cop.feedback(al.id, confirmed=False)
    assert fb["bias_offset"] < 0


@pytest.mark.skipif(not os.path.exists("models/yolo11n-pose.onnx"), reason="pose model not downloaded")
def test_pose_model_detects_people():
    import ultralytics
    from edgecopilot.perception.detector import OnnxYolo
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    m = OnnxYolo("models/yolo11n-pose.onnx", ["person"], pose=True, conf=0.4)
    out = m(img)
    assert len(out.detections) >= 3 and out.feature is not None
