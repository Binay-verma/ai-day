"""API security and behaviour tests (no models or cameras needed)."""
import time

import pytest
from fastapi.testclient import TestClient

from edgecopilot.config_schema import load_config_dict
from edgecopilot.security import AuthManager, hash_password, key_digest
from edgecopilot.server.app import create_app
from edgecopilot.service import EdgeService
from edgecopilot.storage import Store

USERS = [{"username": "asha", "role": "operator", "password_hash": hash_password("operator pass 1")},
         {"username": "vik", "role": "viewer", "password_hash": hash_password("viewer pass 12")},
         {"username": "root", "role": "admin", "password_hash": hash_password("admin pass 123")}]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGECOPILOT_METRICS_TOKEN", "metrics-token-abc")
    cfg = load_config_dict("configs/campus.yaml")
    for k in cfg["models"]:
        cfg["models"][k]["path"] = "models/__missing__.onnx"
    cfg["server"]["rate_limit_per_min"] = 10000
    store = Store(str(tmp_path / "db.sqlite"), str(tmp_path / "media"))
    svc = EdgeService(cfg, root=str(tmp_path), store=store, start_mqtt=False)
    auth = AuthManager(USERS, b"s" * 48, sensor_key_digests=[key_digest("esp32-key-1")])
    app = create_app(svc, auth, cfg)
    with TestClient(app) as c:
        yield c, svc
    store.close()


def login(c, u, p):
    r = c.post("/api/login", json={"username": u, "password": p})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def seed_alert(svc, status="open"):
    rec = {"id": "a1", "camera_id": "cam1", "zone": "Corridor B", "hazard": "fall", "subject": "person#1",
           "t_video": 1.0, "wall_time": time.time(), "prob": 0.9, "peak_prob": 0.9, "severity": "HIGH", "triage": "ALARM",
           "status": status, "explanation": {"text": "<script>alert(1)</script>", "headline": "Fall detected", "because": "b",
                                             "confidence_text": "Confidence 90% (threshold 60%)."},
           "decision": {"counterfactuals": [], "sufficient_reason": ["torso_angle"], "margin_logit": 1.8},
           "verification": {"groundedness": 1.0, "n_claims": 3}, "recommendations": [{"text": "Send first aid", "priority": 1}],
           "cues": [{"name": "torso_angle", "value": 80, "unit": "deg", "contribution": 3.0, "reliability": 1.0,
                     "phrase": "torso tilted 80 deg", "source": "vision"}]}
    rec["snapshot_path"] = svc.store.put_media("a1_snapshot.jpg", b"\xff\xd8jpegdata")
    svc.store.save_alert(rec)


def test_everything_requires_login(env):
    c, _ = env
    assert c.get("/healthz").status_code == 200
    assert c.get("/").status_code == 200
    for path in ("/api/status", "/api/alerts", "/api/cameras", "/api/me", "/api/stats", "/api/audit", "/metrics"):
        assert c.get(path).status_code == 401, path
    assert c.post("/api/alerts/a1/feedback", json={"confirmed": True}).status_code == 401


def test_security_headers(env):
    c, _ = env
    r = c.get("/")
    assert r.headers["x-frame-options"] == "DENY"
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert "unsafe-inline" not in r.headers["content-security-policy"].split("script-src")[1].split(";")[0]


def test_roles_are_enforced(env):
    c, svc = env
    seed_alert(svc)
    v = login(c, "vik", "viewer pass 12")
    h = {"Authorization": f"Bearer {v}"}
    assert c.get("/api/alerts", headers=h).status_code == 200
    assert c.post("/api/alerts/a1/feedback", json={"confirmed": False}, headers=h).status_code == 403
    assert c.get("/api/audit", headers=h).status_code == 403
    o = {"Authorization": f"Bearer {login(c, 'asha', 'operator pass 1')}"}
    r = c.post("/api/alerts/a1/feedback", json={"confirmed": False, "note": "cleaning staff"}, headers=o)
    assert r.status_code == 200 and r.json()["status"] == "dismissed"
    assert svc.store.get_alert("a1")["operator"] == "asha"
    assert c.get("/api/audit", headers=o).status_code == 403
    adm = {"Authorization": f"Bearer {login(c, 'root', 'admin pass 123')}"}
    log = c.get("/api/audit", headers=adm).json()
    assert any(e["action"] == "alert_dismissed" and e["actor"] == "asha" for e in log)
    assert c.get("/api/audit/verify", headers=adm).json()["ok"]


def test_alert_detail_report_and_media(env):
    c, svc = env
    seed_alert(svc)
    o = {"Authorization": f"Bearer {login(c, 'asha', 'operator pass 1')}"}
    d = c.get("/api/alerts/a1", headers=o).json()
    assert d["snapshot_jpg_b64"] and "snapshot_path" not in d  # server paths are never exposed
    rep = c.get("/api/alerts/a1/report.html", headers=o).text
    assert "<script>alert(1)</script>" not in rep and "&lt;script&gt;" in rep  # escaped
    assert c.get("/api/alerts/" + "x" * 100, headers=o).status_code == 404
    assert "torso" in c.post("/api/alerts/a1/ask", json={"question": "why?"}, headers=o).json()["answer"]


def test_cookie_session_and_logout(env):
    c, _ = env
    login(c, "vik", "viewer pass 12")  # TestClient keeps the HttpOnly cookie
    assert c.get("/api/me").json()["username"] == "vik"
    assert c.post("/api/logout").status_code == 200
    assert c.get("/api/me").status_code == 401


def test_bad_login_and_lockout(env):
    c, svc = env
    for _ in range(5):
        assert c.post("/api/login", json={"username": "asha", "password": "wrong"}).status_code == 401
    assert c.post("/api/login", json={"username": "asha", "password": "operator pass 1"}).status_code == 401
    assert any(e["action"] == "login_failed" for e in svc.store.audit_log())


def test_sensor_ingestion_requires_device_key_and_valid_values(env):
    c, svc = env
    assert c.post("/api/sensors", json={"zone": "Lab Block A", "smoke_ppm": 400}).status_code == 401
    k = {"X-Api-Key": "esp32-key-1"}
    assert c.post("/api/sensors", json={"zone": "Lab Block A", "smoke_ppm": 400}, headers=k).status_code == 200
    vals, rel = svc.sensors.cues("Lab Block A", time.time())
    assert vals["smoke_ppm"] == 400 and rel["smoke_ppm"] == 1.0
    for bad in ({"smoke_ppm": -5}, {"smoke_ppm": 1e12}, {"temp_c": "hot"}, {"zone": "x" * 500, "smoke_ppm": 1}, {"zone": "z"}):
        assert c.post("/api/sensors", json=bad, headers=k).status_code == 422, bad
    big = {"zone": "Lab", "smoke_ppm": 1, "pad": "x" * 70000}
    assert c.post("/api/sensors", json=big, headers=k).status_code == 413


def test_metrics_token(env):
    c, _ = env
    assert c.get("/metrics", headers={"Authorization": "Bearer metrics-token-abc"}).status_code == 200
    assert c.get("/metrics", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_websocket_requires_auth(env):
    c, _ = env
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect), c.websocket_connect("/ws") as ws:
        ws.receive_text()
    tok = login(c, "vik", "viewer pass 12")
    with c.websocket_connect(f"/ws?token={tok}") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["role"] == "viewer"


def test_legal_hold_and_erasure(env):
    c, svc = env
    seed_alert(svc)
    adm = {"Authorization": f"Bearer {login(c, 'root', 'admin pass 123')}"}
    op = {"Authorization": f"Bearer {login(c, 'asha', 'operator pass 1')}"}
    assert c.post("/api/alerts/a1/erase", headers=op).status_code == 403
    assert c.post("/api/alerts/a1/hold", json={"hold": True}, headers=adm).json()["legal_hold"]
    assert c.post("/api/alerts/a1/erase", headers=adm).status_code == 409  # held incidents cannot be erased
    assert svc.store.purge(0.0)["incidents"] == 0  # nor purged
    c.post("/api/alerts/a1/hold", json={"hold": False}, headers=adm)
    r = c.post("/api/alerts/a1/erase", headers=adm).json()
    assert r["files"] == 1 and svc.store.get_alert("a1")["subject"] == "redacted"
    assert c.get("/api/alerts/a1", headers=adm).json()["snapshot_jpg_b64"] == ""
    assert c.get("/api/alerts", headers=adm).json()[0]["time_utc"].endswith("Z")


def test_camera_fault_is_a_trouble_alert_and_deduplicated(env):
    c, svc = env
    sent = []
    svc.notifier.alert = lambda rec: sent.append(rec)
    svc._on_fault("cam9", "blackout", "camera cam9: blackout detected")
    svc._on_fault("cam9", "blackout", "camera cam9: blackout detected")  # within cooldown
    rows = svc.store.list_alerts(hazard="camera_fault")
    assert len(rows) == 1 and rows[0]["triage"] == "TROUBLE" and len(sent) == 1
    assert "NOT being monitored" in rows[0]["explanation"]["text"]


def test_console_mute_is_audited(env):
    c, svc = env
    h = {"Authorization": f"Bearer {login(c, 'asha', 'operator pass 1')}"}
    assert c.post("/api/console/sound", json={"on": False}, headers=h).status_code == 200
    acts = [r["action"] for r in svc.store.audit_log(20)]
    assert "console_muted" in acts
    c.cookies.clear()
    assert c.post("/api/console/sound", json={"on": True}).status_code == 401


def test_feedback_metric_is_per_camera_and_hazard(env):
    from edgecopilot.server import metrics as prom
    c, svc = env
    seed_alert(svc)
    h = {"Authorization": f"Bearer {login(c, 'asha', 'operator pass 1')}"}
    assert c.post("/api/alerts/a1/feedback", json={"confirmed": False}, headers=h).status_code == 200
    prom.refresh(svc)
    body = c.get("/metrics", headers={"Authorization": "Bearer metrics-token-abc"}).text
    assert 'edgecopilot_operator_feedback_total{camera="cam1",hazard="fall",outcome="dismissed"}' in body
    assert "edgecopilot_frame_latency_seconds" in body


def test_metrics_refresh_with_a_camera(env, tmp_path):
    """refresh() runs inside a catch-all in production, so exercise it directly."""
    from edgecopilot.cameras import CameraWorker
    from edgecopilot.pipeline import Copilot
    from edgecopilot.server import metrics as prom
    c, svc = env
    cfg = load_config_dict("configs/campus.yaml")
    for k in cfg["models"]:
        cfg["models"][k]["path"] = "models/__missing__.onnx"
    w = CameraWorker("camX", "missing.mp4", Copilot(cfg, root=str(tmp_path), zone="Corridor B"), realtime=False)
    w.health.last_frame_wall = time.time() - 3
    w.health.frames = 5
    w.latency_ms.extend([40, 50, 60, 70, 80])
    svc.cameras["camX"] = w
    prom.refresh(svc)
    prom.refresh(svc)  # second refresh must not re-observe the same frames
    body = prom.render().decode()
    assert 'edgecopilot_frame_latency_seconds_count{camera="camX"} 5.0' in body
    assert 'edgecopilot_alert_threshold{camera="camX",hazard="fall"} 0.6' in body
    assert 'edgecopilot_last_frame_age_seconds{camera="camX"}' in body
    del svc.cameras["camX"]


def test_what_if_rescoring_is_exact(env, tmp_path):
    import math

    from edgecopilot.pipeline import Copilot
    c, svc = env
    cfg = load_config_dict("configs/campus.yaml")
    for k in cfg["models"]:
        cfg["models"][k]["path"] = "models/__missing__.onnx"
    cop = Copilot(cfg, root=str(tmp_path), zone="Corridor B")
    svc.copilots["cam1"] = cop
    seed_alert(svc)
    h = {"Authorization": f"Bearer {login(c, 'vik', 'viewer pass 12')}"}
    r = c.post("/api/alerts/a1/whatif", json={"values": {}}, headers=h).json()
    assert abs(r["prob"] - 0.9) < 1e-9 and r["would_alert"]
    spec = cop.models["fall"].specs["torso_angle"]
    r = c.post("/api/alerts/a1/whatif", json={"values": {"torso_angle": spec.ref}}, headers=h).json()
    expect = math.log(0.9 / 0.1) - spec.weight * (80 - spec.ref)
    assert abs(math.log(r["prob"] / (1 - r["prob"])) - expect) < 1e-9
    assert r["would_alert"] == (expect >= math.log(0.9 / 0.1) - 1.8)
    assert c.post("/api/alerts/a1/whatif", json={"values": {"x": 1}}).status_code in (200, 401)


def test_self_test_flags_degraded_camera_and_sends_test(env, tmp_path):
    from edgecopilot.cameras import CameraWorker
    from edgecopilot.pipeline import Copilot
    c, svc = env
    cfg = load_config_dict("configs/campus.yaml")
    for k in cfg["models"]:
        cfg["models"][k]["path"] = "models/__missing__.onnx"
    w = CameraWorker("camT", "missing.mp4", Copilot(cfg, root=str(tmp_path), zone="Corridor B"), realtime=False)
    svc.cameras["camT"] = w
    adm = {"Authorization": f"Bearer {login(c, 'root', 'admin pass 123')}"}
    vw = {"Authorization": f"Bearer {login(c, 'vik', 'viewer pass 12')}"}
    assert c.post("/api/selftest", json={}, headers=vw).status_code == 403
    sent = []
    svc.notifier.alert = lambda rec: sent.append(rec)
    r = c.post("/api/selftest", json={"notify": True}, headers=adm).json()
    cam = r["cameras"]["camT"]
    assert cam["checks"]["alert_chain"]["ok"], cam          # fusion -> explanation -> verifier -> playbook works
    assert not cam["checks"]["pose_model"]["ok"]           # no model loaded = degraded = flagged
    assert not r["ok"]
    assert any(x["hazard"] == "camera_fault" and x["subject"] == "self_test_failed" for x in sent)
    assert any(x["hazard"] == "self_test" and "TEST" in x["explanation"]["text"] for x in sent)
    assert c.get("/api/selftest", headers=vw).json()["ok"] is False
    del svc.cameras["camT"]


def test_long_lie_escalation_is_persisted_audited_and_notified(env):
    from types import SimpleNamespace
    c, svc = env
    seed_alert(svc)
    sent = []
    svc.notifier.alert = lambda rec: sent.append(rec)
    svc._on_status(SimpleNamespace(id="a1", zone="Corridor B", subject="person#1", peak_prob=0.9), "long_lie")
    a = svc.store.get_alert("a1")
    assert a["severity"] == "CRITICAL" and a["triage"] == "ALARM"
    assert sent and sent[0]["explanation"]["headline"].startswith("STILL DOWN")
    assert any(r["action"] == "long_lie_escalation" for r in svc.store.audit_log(10))


def test_incident_correlation_and_handover(env):
    c, svc = env
    now = time.time()

    def rec(i, cam, zone, hz, t, status="open", sev="HIGH"):
        return {"id": i, "camera_id": cam, "zone": zone, "hazard": hz, "subject": "s", "t_video": 0.0, "wall_time": t,
                "prob": 0.9, "peak_prob": 0.9, "severity": sev, "triage": "ALARM", "status": status,
                "explanation": {"headline": f"{hz} in {zone}", "text": "x"}, "decision": {}, "verification": {},
                "recommendations": [], "cues": []}
    svc.store.save_alert(rec("f1", "cam1", "Lab A", "fire_smoke", now - 300))
    svc.store.save_alert(rec("c1", "cam2", "Lab A", "crowd", now - 260, sev="CRITICAL"))   # same incident, other camera
    svc.store.save_alert(rec("x1", "cam3", "Hall", "fall", now - 250, status="dismissed"))
    svc.store.save_alert(rec("f2", "cam1", "Lab A", "fire_smoke", now - 3000, status="resolved"))  # separate (time gap)
    h = {"Authorization": f"Bearer {login(c, 'asha', 'operator pass 1')}"}
    inc = c.get("/api/incidents?hours=2", headers=h).json()
    lab = [i for i in inc if i["zone"] == "Lab A"]
    assert len(lab) == 2
    big = max(lab, key=lambda i: i["n_alerts"])
    assert big["n_alerts"] == 2 and big["multi_signal"] and big["severity"] == "CRITICAL" and big["open"]
    d = c.get("/api/handover?hours=2", headers=h).json()
    assert len(d["open_incidents"]) == 1 and d["dismissed"][0]["id"] == "x1"
    page = c.get("/api/handover.html?hours=2", headers=h)
    assert page.status_code == 200 and "Open incidents" in page.text and "<script" not in page.text
    assert any(r["action"] == "handover_exported" for r in svc.store.audit_log(5))
