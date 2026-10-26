"""'Use my camera': an operator's browser camera analysed over /ws/camera (no models needed)."""
import json
import struct
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from edgecopilot.cameras import BrowserCamera
from edgecopilot.config_schema import load_config_dict
from edgecopilot.security import AuthManager
from edgecopilot.server.app import create_app, decode_camera_frame
from edgecopilot.service import EdgeService
from edgecopilot.storage import Store
from test_server import USERS, login


def make_env(tmp_path, **bc):
    cfg = load_config_dict("configs/campus.yaml")
    for k in cfg["models"]:
        cfg["models"][k]["path"] = "models/__missing__.onnx"
    cfg["server"]["rate_limit_per_min"] = 10000
    cfg["server"]["browser_camera"].update(bc)
    store = Store(str(tmp_path / "db.sqlite"), str(tmp_path / "media"))
    svc = EdgeService(cfg, root=str(tmp_path), store=store, start_mqtt=False)
    return svc, store, create_app(svc, AuthManager(USERS, b"s" * 48), cfg)


@pytest.fixture()
def env(tmp_path):
    svc, store, app = make_env(tmp_path)
    with TestClient(app) as c:
        yield c, svc
    store.close()


def frame_msg(t=0.0, w=320, h=240, value=90):
    img = np.full((h, w, 3), value, np.uint8)
    cv2.rectangle(img, (40, 40), (120, 200), (200, 180, 160), -1)
    ok, buf = cv2.imencode(".jpg", img)
    return struct.pack("<d", t) + buf.tobytes()


def recv(ws):
    return json.loads(ws.receive_text())


def test_login_and_operator_role_required(env):
    c, svc = env
    with c.websocket_connect("/ws/camera") as ws:
        assert recv(ws)["code"] == "login"
    tok = login(c, "vik", "viewer pass 12")
    with c.websocket_connect(f"/ws/camera?token={tok}") as ws:
        assert recv(ws)["code"] == "role"
    assert not svc.browser_cams


def test_foreign_origin_refused(env):
    c, _ = env
    tok = login(c, "asha", "operator pass 1")
    with pytest.raises(WebSocketDisconnect), c.websocket_connect(f"/ws/camera?token={tok}",
                                                                 headers={"Origin": "https://evil.example"}) as ws:
        ws.receive_text()


def test_session_round_trip_is_private_and_audited(env):
    c, svc = env
    tok = login(c, "asha", "operator pass 1")
    h = {"Authorization": f"Bearer {tok}"}
    me = c.get("/api/me", headers=h).json()["browser_camera"]
    assert me["enabled"] and me["in_use"] == 0
    with c.websocket_connect(f"/ws/camera?token={tok}") as ws:
        ready = recv(ws)
        assert ready["type"] == "ready" and ready["camera"].startswith("browser-") and ready["zone"]
        cam = ready["camera"]
        assert cam in svc.browser_cams and svc.browser_cams[cam].owner == "asha"
        for i in range(3):
            ws.send_bytes(frame_msg(t=100.0 + 0.2 * i))
            m = recv(ws)
            assert m["type"] in ("frame", "skip")
            if m["type"] == "frame":
                assert m["camera"] == cam and m["jpg"] and "gauges" in m["state"]
            time.sleep(0.12)
        # not a supervised camera: not listed, no frame endpoint, no self-test, health unaffected
        assert all(x["id"] != cam for x in c.get("/api/cameras", headers=h).json())
        assert c.get(f"/api/cameras/{cam}/frame.jpg", headers=h).status_code == 404
        assert cam not in svc.self_test()["cameras"] and svc.healthy()
        assert c.get("/api/me", headers=h).json()["browser_camera"]["in_use"] == 1
        ws.send_text('{"type": "stop"}')
    for _ in range(50):
        if not svc.browser_cams:
            break
        time.sleep(0.05)
    assert not svc.browser_cams and cam not in svc.cameras
    actions = [(a["action"], a["target"]) for a in svc.store.audit_log(50)]
    assert ("browser_camera_started", cam) in actions and ("browser_camera_stopped", cam) in actions


def test_bad_frames_are_rejected_without_ending_the_session(env):
    c, _ = env
    tok = login(c, "asha", "operator pass 1")
    with c.websocket_connect(f"/ws/camera?token={tok}") as ws:
        assert recv(ws)["type"] == "ready"
        ws.send_bytes(struct.pack("<d", 1.0) + b"PNG not a jpeg at all")
        assert recv(ws)["code"] == "bad_frame"
        time.sleep(0.12)
        ws.send_bytes(frame_msg(1.2))
        assert recv(ws)["type"] == "frame"


def test_slots_are_limited(tmp_path):
    svc, store, app = make_env(tmp_path, max_sessions=1)
    with TestClient(app) as c:
        tok = login(c, "asha", "operator pass 1")
        with c.websocket_connect(f"/ws/camera?token={tok}") as a:
            assert recv(a)["type"] == "ready"
            with c.websocket_connect(f"/ws/camera?token={tok}") as b:
                assert recv(b)["code"] == "busy"
    store.close()


def test_disabled_and_time_limit(tmp_path):
    svc, store, app = make_env(tmp_path, enabled=False)
    with TestClient(app) as c:
        tok = login(c, "asha", "operator pass 1")
        assert not c.get("/api/me", headers={"Authorization": f"Bearer {tok}"}).json()["browser_camera"]["enabled"]
        with c.websocket_connect(f"/ws/camera?token={tok}") as ws:
            assert recv(ws)["code"] == "disabled"
    store.close()
    svc, store, app = make_env(tmp_path / "b", max_minutes=0.002)  # 0.12 s
    with TestClient(app) as c:
        tok = login(c, "asha", "operator pass 1")
        with c.websocket_connect(f"/ws/camera?token={tok}") as ws:
            assert recv(ws)["type"] == "ready"
            time.sleep(0.2)
            ws.send_bytes(frame_msg(0.0))
            assert recv(ws)["code"] == "time_limit"
    store.close()


def test_decode_camera_frame_limits():
    img, t = decode_camera_frame(frame_msg(t=12.5))
    assert img is not None and img.shape == (240, 320, 3) and t == 12.5
    assert decode_camera_frame(b"")[0] is None
    assert decode_camera_frame(struct.pack("<d", 1.0) + b"\xff\xd8" + b"\0" * 2_000_000)[0] is None  # oversized
    assert decode_camera_frame(frame_msg(w=32, h=32))[0] is None                                  # too small
    assert decode_camera_frame(struct.pack("<d", float("nan")) + frame_msg()[8:])[1] is None      # bad clock: server time


def test_stream_time_is_monotonic_and_cannot_run_ahead():
    class Cop:  # stream_time does not touch the copilot
        zone = "Corridor B"
    w = BrowserCamera("browser-x", Cop(), owner="asha")
    t0 = w.stream_time(500.0)
    assert t0 == pytest.approx(0.0, abs=0.05)
    assert w.stream_time(500.0) > t0                      # repeated capture time still moves forward
    assert w.stream_time(400.0) > t0                      # a clock going backwards cannot rewind the stream
    assert w.stream_time(10_000.0) < 2.0                  # nor fast-forward it beyond the server's clock


def test_demo_reel_yields_cpu_but_live_cameras_never_do(tmp_path):
    svc, store, app = make_env(tmp_path, background_fps=2.0)
    reel = svc.add_camera({"id": "reel", "source": "demo/demo_campus.mp4", "zone": "Corridor B"})
    live = svc.add_camera({"id": "door", "source": "rtsp://cam.local/stream", "zone": "Corridor B"})
    w = svc.add_browser_camera("asha")
    assert reel.throttle_fps == 2.0 and live.throttle_fps == 0.0
    assert reel.status()["throttled_fps"] == 2.0 and live.status()["throttled_fps"] is None
    assert w.copilot.enable["crowd"] is False and w.copilot.fire_every == 3   # fall + fire only, fire every 3rd frame
    svc.remove_browser_camera(w.cam_id)
    assert reel.throttle_fps == 0.0
    store.close()

