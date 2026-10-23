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
