"""Perception unit tests with synthetic, physically meaningful inputs (no models needed)."""
import os

import cv2
import numpy as np
import pytest

from edgecopilot.io.video import VideoSource
from edgecopilot.perception.detector import Detection, letterbox, nms
from edgecopilot.perception.fall import FallFeatureExtractor, torso_angle
from edgecopilot.perception.tracker import Tracker, iou_matrix
from edgecopilot.privacy import anonymize
from edgecopilot.xai.saliency import box_renormalize, eigencam, overlay, skeleton_saliency


def person(cx, top, h, angle_deg=0.0, conf=0.9):
    """COCO-17 key-points of a stick person of height h, torso rotated by angle (deg from vertical)."""
    kp = np.zeros((17, 3))
    a = np.radians(angle_deg)
    hip = np.array([cx, top + 0.55 * h])
    up = np.array([np.sin(a), -np.cos(a)])
    sh = hip + up * 0.35 * h
    head = hip + up * 0.5 * h
    for i, p in ((0, head), (1, head), (2, head), (5, sh - [4, 0]), (6, sh + [4, 0]), (11, hip - [4, 0]), (12, hip + [4, 0]),
                 (15, hip - up * 0.45 * h), (16, hip - up * 0.45 * h)):
        kp[i] = [p[0], p[1], conf]
    pts = kp[kp[:, 2] > 0][:, :2]
    box = np.array([pts[:, 0].min() - 8, pts[:, 1].min() - 8, pts[:, 0].max() + 8, pts[:, 1].max() + 8])
    return Detection(box, conf, 0, "person", kp)


def test_letterbox_and_nms():
    img = np.zeros((480, 640, 3), np.uint8)
    out, s, px, py = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and s == 1.0 and px == 0 and py == 80
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert sorted(keep.tolist()) == [0, 2]
    assert len(nms(np.zeros((0, 4)), np.zeros(0), 0.5)) == 0


def test_iou_and_tracker_identity():
    assert iou_matrix(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]]))[0, 0] == pytest.approx(1.0)
    tr = Tracker()
    ids = []
    for i in range(10):
        d = [person(100 + 3 * i, 50, 200), person(400, 60, 180)]
        tr.update(d, i / 10)
        ids.append((d[0].track_id, d[1].track_id))
    assert len(set(ids)) == 1 and ids[0][0] != ids[0][1]


def test_tracker_low_confidence_only_continues():
    tr = Tracker(high_thr=0.35)
    tr.update([person(100, 50, 200, conf=0.2)], 0.0)
    assert not tr.tracks  # low-confidence detection cannot start a track
    d = person(100, 50, 200, conf=0.9)
    tr.update([d], 0.1)
    low = person(102, 52, 200, conf=0.2)
    tr.update([low], 0.2)
    assert low.track_id == d.track_id  # but it keeps an existing one alive


def test_torso_angle():
    assert torso_angle(person(100, 50, 200, 0).keypoints) == pytest.approx(0, abs=1)
    assert torso_angle(person(100, 50, 200, 90).keypoints) == pytest.approx(90, abs=1)


def simulate(angle_fn, top_fn, n=60, fps=15.0):
    tr, fx = Tracker(), FallFeatureExtractor()
    vals = []
    for i in range(n):
        t = i / fps
        d = person(200, top_fn(t), 200, angle_fn(t))
        for track in tr.update([d], t):
            if track.hits >= 2:
                vals.append(fx(track, t)[0])
    return vals


def test_fall_vs_lying_down_cues():
    # fall: upright, then within 0.5 s rotate to horizontal and drop
    fall = simulate(lambda t: 0 if t < 1 else min(90, (t - 1) * 180), lambda t: 50 if t < 1 else 50 + min(1, (t - 1) * 2) * 120)
    # lying down slowly over 3 s
    lie = simulate(lambda t: 0 if t < 1 else min(90, (t - 1) * 30), lambda t: 50 if t < 1 else 50 + min(1, (t - 1) / 3) * 120, n=90)
    f, li = fall[-1], lie[-1]
    assert f["torso_angle"] > 70 and li["torso_angle"] > 70
    assert f["descent_speed"] > 2 * li["descent_speed"]
    assert f["tilt_rate"] > 2 * li["tilt_rate"]
    assert f["ground_time"] > 0.5


def test_saliency_maps():
    feat = np.zeros((16, 20, 20), np.float32)
    feat[:, 5:9, 12:16] = np.random.default_rng(0).random((16, 4, 4)) + 2.0
    cam = eigencam(feat, (1.0, 0, 0, 160, 160), (160, 160))
    y, x = np.unravel_index(cam.argmax(), cam.shape)
    assert 40 <= y < 72 and 96 <= x < 128  # peak where the activation is
    rn = box_renormalize(cam, [np.array([0, 0, 50, 50])])
    assert rn[:50, :50].max() <= 1.0 and rn[60:, 60:].max() == 0
    sk = skeleton_saliency((200, 200), [person(100, 20, 150).keypoints])
    assert sk.max() == pytest.approx(1.0)
    ov = overlay(np.zeros((200, 200, 3), np.uint8), sk)
    assert ov.shape == (200, 200, 3)


def test_privacy_skeleton_mode_removes_detail():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    out = anonymize(img, [person(160, 20, 200)], "skeleton")
    # high-frequency content (faces, text, texture) is gone
    assert cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < \
        0.2 * cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert anonymize(img, [], "raw") is img


def test_video_source_stride_and_time(tmp_path):
    p = str(tmp_path / "v.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(30):
        vw.write(np.full((48, 64, 3), i * 8, np.uint8))
    vw.release()
    frames = list(VideoSource(p, stride=3))
    assert len(frames) == 10 and frames[1][1] == pytest.approx(0.3, abs=1e-6)
    assert len(list(VideoSource(p, max_frames=4))) == 4
    with pytest.raises(OSError):
        VideoSource(str(tmp_path / "missing.mp4"))


@pytest.mark.skipif(not os.path.exists("models/yolo11n-pose.onnx"), reason="models not downloaded")
def test_service_end_to_end_on_video(tmp_path):
    """Camera worker -> copilot -> store -> clip, on a synthetic smoke video, headless."""
    import time
    from edgecopilot.config_schema import load_config_dict
    from edgecopilot.service import EdgeService
    from edgecopilot.storage import Store
    p = str(tmp_path / "smoke.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for i in range(200):
        f = base.copy()
        if i > 40:
            a = min(0.85, (i - 40) / 60)
            f[:150] = (f[:150] * (1 - a) + 200 * a).astype(np.uint8)
            f[150:200, 140:180] = (20, 100 + (i % 3) * 60, 255)
        vw.write(f)
    vw.release()
    cfg = load_config_dict("configs/campus.yaml")
    cfg["storage"]["clip_post_s"] = 1.0
    store = Store(str(tmp_path / "db.sqlite"), str(tmp_path / "media"))
    svc = EdgeService(cfg, root=os.getcwd(), store=store, start_mqtt=False)
    from edgecopilot.io.sensors import synthetic_fire_trace
    for r in synthetic_fire_trace("Lab Block A", 0, 20, fire_at=4):  # MQ-2 + temperature in the same room
        svc.sensors.push(r)
    got = []
    svc.listeners.append(lambda m: got.append(m["type"]))
    w = svc.add_camera({"id": "t1", "source": p, "zone": "Lab Block A", "realtime": False, "hazards": {"fall": False, "crowd": False}})
    svc.start()
    w.thread.join(120)
    time.sleep(1.5)
    rows = store.list_alerts()
    assert rows and rows[0]["hazard"] == "fire_smoke" and rows[0]["camera_id"] == "t1"
    assert rows[0]["provenance"]["models"]["pose"] != "missing"
    assert "frame" in got and "alert" in got
    assert store.verify_audit_chain()["ok"]
    svc.stop()


def test_zone_rules_dwell_hours_and_gap():
    import numpy as np

    from edgecopilot.perception.detector import Detection
    from edgecopilot.perception.zones import ZoneRules, in_window
    assert in_window((20 * 60, 7 * 60), 23 * 60) and in_window((20 * 60, 7 * 60), 3 * 60)
    assert not in_window((20 * 60, 7 * 60), 12 * 60) and in_window(None, 12 * 60)
    z = ZoneRules([{"name": "store", "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]}])
    inside = Detection(np.array([350.0, 100.0, 400.0, 300.0]), 0.9, 0, "person")   # foot point (375, 300) in right half
    outside = Detection(np.array([50.0, 100.0, 100.0, 300.0]), 0.9, 0, "person")
    assert z([outside], (480, 640), 0.0)["store"]["dwell_time"] == 0.0
    v = None
    for k in range(11):
        v = z([inside], (480, 640), float(k))["store"]
    assert v["dwell_time"] == 10.0 and v["people"] == 1 and v["after_hours"] == 1.0
    v = z([], (480, 640), 11.5)["store"]           # a short gap (occlusion) is bridged
    assert v["after_hours"] == 0.0
    v = z([inside], (480, 640), 12.0)["store"]
    assert v["dwell_time"] == 12.0
    z([], (480, 640), 20.0)
    assert z([inside], (480, 640), 21.0)["store"]["dwell_time"] == 0.0  # a long absence restarts the clock


def test_intrusion_alert_end_to_end(tmp_path):
    import numpy as np

    from edgecopilot.config_schema import load_config_dict
    from edgecopilot.perception.detector import Detection, DetectorOutput
    from edgecopilot.pipeline import Copilot

    class FakePose:
        def __call__(self, img):
            kp = np.zeros((17, 3), np.float32)
            return DetectorOutput([Detection(np.array([400.0, 100.0, 460.0, 400.0]), 0.9, 0, "person", kp)], None, (1, 0, 0, 640, 640), 1.0)
    cfg = load_config_dict("configs/campus.yaml")
    cfg["models"]["fire_smoke"]["path"] = "models/__missing__.onnx"
    cop = Copilot(cfg, root=str(tmp_path), zone="Lab Block A", pose_model=FakePose(),
                  enable={"fall": False, "crowd": False, "fire_smoke": False},
                  regions=[{"name": "Chemical store", "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]}])
    alerts = []
    for k in range(400):
        frame = np.full((480, 640, 3), 100 + (k % 2), np.uint8)
        alerts += cop.process(frame, k / 10.0).new_alerts
    assert len(alerts) == 1 and alerts[0].hazard == "intrusion"
    al = alerts[0]
    assert al.verification["groundedness"] == 1.0 and "restricted area" in al.explanation["text"]
    assert any("patrol" in r["text"] for r in al.recommendations)
