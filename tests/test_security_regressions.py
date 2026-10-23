"""Regression tests for the independent security audit (docs/research/audit_security.md, findings S1-S13)."""
import json
import os
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from edgecopilot.cameras import redact
from edgecopilot.config_schema import load_config_dict
from edgecopilot.io.mqtt import parse_sensor_payload
from edgecopilot.io.sensors import SensorHub
from edgecopilot.security import AuthManager, hash_password, key_digest
from edgecopilot.server.app import create_app
from edgecopilot.service import EdgeService
from edgecopilot.storage import Store

USERS = [{"username": "asha", "role": "operator", "password_hash": hash_password("operator pass 1")},
         {"username": "root", "role": "admin", "password_hash": hash_password("admin pass 123")}]
DEVICES = [{"id": "esp-lab", "digest": key_digest("lab-key"), "zones": ["Lab Block A"]}]


@pytest.fixture()
def env(tmp_path):
    cfg = load_config_dict("configs/campus.yaml")
    for k in cfg["models"]:
        cfg["models"][k]["path"] = "models/__missing__.onnx"
    cfg["server"]["rate_limit_per_min"] = 100000
    store = Store(str(tmp_path / "db.sqlite"), str(tmp_path / "media"), audit_key=b"k" * 32)
    svc = EdgeService(cfg, root=str(tmp_path), store=store, start_mqtt=False)
    auth = AuthManager(USERS, b"s" * 48, sensor_devices=DEVICES)
    with TestClient(create_app(svc, auth, cfg)) as c:
        yield c, svc, auth
    store.close()


def bearer(c, u, p):
    return {"Authorization": "Bearer " + c.post("/api/login", json={"username": u, "password": p}).json()["token"]}


# S1 - concurrent sensor pushes must not break readers; a frame error must not kill the camera thread
def test_s1_sensor_hub_is_thread_safe():
    hub = SensorHub(max_zones=10000)
    errors = []

    def writer(k):
        for i in range(3000):
            hub.push({"zone": f"z{k}-{i % 500}", "t": float(i), "smoke_ppm": 1.0})

    def reader():
        try:
            for _ in range(3000):
                hub.status(1.0)
                hub.cues("z0-1", 1.0)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    ts = [threading.Thread(target=writer, args=(k,)) for k in range(4)] + [threading.Thread(target=reader)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors


def test_s1_camera_survives_processing_errors(tmp_path):
    import cv2
    from edgecopilot.cameras import CameraWorker
    p = str(tmp_path / "v.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(20):
        vw.write(np.full((48, 64, 3), i * 10, np.uint8))
    vw.release()

    class Boom:
        zone, privacy = "z", "blur"
        sensors = SensorHub()
        _last_frame = None

        def reset(self):
            pass

        def process(self, frame, t):
            raise RuntimeError("bad frame")
    faults = []
    w = CameraWorker("c1", p, Boom(), realtime=False, on_fault=lambda *a: faults.append(a[1]))
    w.start()
    w.thread.join(10)
    assert w.health.errors == 20 and w.health.frames == 20
    assert faults == ["processing_error", "processing_error"]  # reported at the 1st and 10th error, not every frame


def test_s1_liveness_detects_dead_camera_thread(env):
    c, svc, _ = env

    class W:
        class H:
            state, last_frame_wall = "live", time.time()
        health = H()
        paused = False

        class T:
            @staticmethod
            def is_alive():
                return False
        thread = T()

        def stop(self):
            pass
    svc.cameras["dead"] = W()
    r = c.get("/healthz")
    assert r.status_code == 503 and "died" in r.json()["problem"]
    del svc.cameras["dead"]


# S2 - device keys are bound to zones; MQTT topic decides the zone
def test_s2_device_key_bound_to_zone(env):
    c, svc, _ = env
    k = {"X-Api-Key": "lab-key"}
    assert c.post("/api/sensors", json={"zone": "Lab Block A", "smoke_ppm": 300}, headers=k).status_code == 200
    r = c.post("/api/sensors", json={"zone": "Corridor B", "smoke_ppm": 1400}, headers=k)
    assert r.status_code == 403
    assert "Corridor B" not in svc.sensors.status(time.time())
    assert any(e["action"] == "sensor_zone_denied" for e in svc.store.audit_log())


def test_s2_mqtt_payload_cannot_override_topic_zone():
    r = parse_sensor_payload("campus/Lab_Block_A/sensors", b'{"zone": "Corridor B", "smoke_ppm": 1400}', now=1.0)
    assert r["zone"] == "Lab Block A"


def test_s2_unknown_zones_are_rejected_and_bounded():
    hub = SensorHub(zones=["Lab Block A"])
    assert hub.push({"zone": "Lab Block A", "t": 1.0, "smoke_ppm": 1})
    assert not hub.push({"zone": "made-up", "t": 1.0, "smoke_ppm": 1})
    hub2 = SensorHub(max_zones=5)
    for i in range(100):
        hub2.push({"zone": f"z{i}", "t": 1.0, "smoke_ppm": 1})
    assert len(hub2.hist) == 5


def test_s2_no_cross_room_fallback():
    hub = SensorHub(zones=["Lab Block A", "Corridor B"], fallback_zone=None)
    hub.push({"zone": "Lab Block A", "t": 1.0, "smoke_ppm": 900})
    assert hub.cues("Corridor B", 1.0) == ({}, {})


# S3 - chunked/streamed bodies are limited before auth
def test_s3_chunked_body_limit(env):
    c, _, _ = env

    def gen():
        for _ in range(200):
            yield b"x" * 65536  # 12.5 MB, no Content-Length
    r = c.post("/api/sensors", content=gen(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413


# S4 - credentials never leave the camera module
def test_s4_redact():
    s = "cannot open 'rtsp://admin:S3cret!@10.0.0.5:554/stream'"
    assert "S3cret" not in redact(s) and "rtsp://***@10.0.0.5" in redact(s)


def test_s4_fault_detail_is_redacted(env):
    _, svc, _ = env
    svc.notifier.alert = lambda rec: None
    svc._on_fault("cam1", "offline", "cannot open 'rtsp://u:pw123@10.0.0.9/s'")
    row = svc.store.list_alerts(hazard="camera_fault")[0]
    assert "pw123" not in json.dumps(row) and "pw123" not in json.dumps(svc.store.audit_log())


# S5 - an alert can be labelled once
def test_s5_feedback_only_once(env):
    c, svc, _ = env
    svc.store.save_alert({"id": "f1", "hazard": "fire_smoke", "zone": "Lab Block A", "camera_id": "x", "wall_time": time.time(),
                          "status": "open", "prob": 0.8})
    h = bearer(c, "asha", "operator pass 1")
    assert c.post("/api/alerts/f1/feedback", json={"confirmed": False}, headers=h).status_code == 200
    r = c.post("/api/alerts/f1/feedback", json={"confirmed": False}, headers=h)
    assert r.status_code == 409 and "already dismissed" in r.json()["detail"]


def test_s5_copilot_ignores_repeated_labels(tmp_path):
    from edgecopilot.copilot.events import Alert
    from edgecopilot.pipeline import Copilot
    cfg = load_config_dict("configs/campus.yaml")
    for k in cfg["models"]:
        cfg["models"][k]["path"] = "models/__missing__.onnx"
    cop = Copilot(cfg, root=str(tmp_path), zone="Lab Block A")
    al = Alert("fire_smoke", "Lab Block A", "Lab Block A", 1.0, 0.9, "HIGH", {}, {}, {}, [], [])
    cop.events.add(al)
    assert cop.feedback(al.id, False) is not None
    before = cop.models["fire_smoke"].bias_offset.copy()
    for _ in range(50):
        assert cop.feedback(al.id, False) is None
    assert cop.models["fire_smoke"].bias_offset == before


# S6 - torn calibration file does not stop the service; writes are atomic
def test_s6_corrupt_calibration_file(tmp_path):
    from edgecopilot.pipeline import Copilot
    cfg = load_config_dict("configs/campus.yaml")
    for k in cfg["models"]:
        cfg["models"][k]["path"] = "models/__missing__.onnx"
    os.makedirs(tmp_path / "calibration")
    (tmp_path / "calibration" / "Lab_Block_A.json").write_text('{"fire_smoke": {"alpha": 0.1, "scores": [1.0, 2')
    cop = Copilot(cfg, root=str(tmp_path), zone="Lab Block A")
    assert cop.models["fire_smoke"].threshold == pytest.approx(0.6)
    from edgecopilot.xai.calibration import ConformalGate
    cop.gates["fire_smoke"] = ConformalGate(0.1, scores=[0.0] * 20)
    cop.save_calibration()
    assert json.load(open(cop.calib_path))["fire_smoke"]["n"] == 20
    assert not [f for f in os.listdir(tmp_path / "calibration") if f.endswith(".tmp")]


# S7 - live state carries only the camera's own zone (bounded) - covered by cameras.py; payload size check
def test_s7_status_payload_bounded(env):
    c, svc, _ = env
    for i in range(500):
        svc.sensors.push({"zone": f"spam{i}", "t": time.time(), "smoke_ppm": 1})
    assert len(svc.sensors.hist) == 0  # only configured zones are accepted at all


# S8 - login lockout is per (user, client IP) and does not grow without bound
def test_s8_lockout_isolated_and_bounded():
    a = AuthManager(USERS, b"s" * 48, max_failures=3, lockout_s=60)
    for _ in range(3):
        a.login("asha", "x", "10.0.0.1")
    assert a.login("asha", "operator pass 1", "10.0.0.1") is None       # this client is locked
    assert a.login("asha", "operator pass 1", "192.168.1.9") is not None  # the real operator is not
    now = time.time()
    a._fail.update({(f"u{i}", "1.1.1.1"): [now] for i in range(20000)})  # credential-stuffing residue
    a.locked("x", "y")
    assert len(a._fail) <= 15001


# S10 - NaN never turns a 422 into a 500
def test_s10_nan_is_422(env):
    c, _, _ = env
    r = c.post("/api/sensors", content=b'{"zone": "Lab Block A", "smoke_ppm": NaN}',
               headers={"Content-Type": "application/json", "X-Api-Key": "lab-key"})
    assert r.status_code == 422


# S11 - cross-site WebSocket is refused
def test_s11_websocket_origin(env):
    c, _, _ = env
    from starlette.websockets import WebSocketDisconnect
    tok = bearer(c, "asha", "operator pass 1")["Authorization"][7:]
    with pytest.raises(WebSocketDisconnect), c.websocket_connect(f"/ws?token={tok}", headers={"Origin": "https://evil.example"}) as ws:
        ws.receive_json()
    with c.websocket_connect(f"/ws?token={tok}", headers={"Origin": "http://testserver"}) as ws:
        assert ws.receive_json()["type"] == "hello"


# S12 - keyed audit chain, tail truncation detected
def test_s12_keyed_audit_chain(tmp_path):
    s = Store(str(tmp_path / "a.db"), str(tmp_path / "m"), audit_key=b"secret-key-0123456789abcdef012345")
    for i in range(5):
        s.audit("u", "operator", "act", str(i))
    v = s.verify_audit_chain()
    assert v["ok"] and v["keyed"]
    s.conn.execute("DELETE FROM audit WHERE seq = 5")  # drop the newest entry
    assert not s.verify_audit_chain()["ok"]
    # an attacker without the key cannot recompute a valid chain
    s2 = Store(str(tmp_path / "a.db"), str(tmp_path / "m"), audit_key=b"wrong-key-0123456789abcdef0123456")
    assert not s2.verify_audit_chain()["ok"]


# S13 - API schema needs a login
def test_s13_openapi_private(env):
    c, _, _ = env
    assert c.get("/api/openapi.json").status_code == 401
    assert c.get("/api/openapi.json", headers=bearer(c, "asha", "operator pass 1")).status_code == 200
