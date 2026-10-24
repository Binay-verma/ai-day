"""Tests for the production layer: config validation, auth, storage/audit, notifications, MQTT parsing, camera health."""
import copy
import http.server
import json
import os
import threading
import time

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from edgecopilot.cameras import ClipBuffer, TamperDetector
from edgecopilot.config_schema import Config, load_validated
from edgecopilot.io.mqtt import parse_sensor_payload
from edgecopilot.notify import Notifier, WebhookChannel
from edgecopilot.security import AuthManager, RateLimiter, hash_password, key_digest, verify_password
from edgecopilot.storage import Store

RAW = yaml.safe_load(open("configs/campus.yaml"))


# ------------------------------------------------------------------ config
def test_shipped_config_is_valid():
    cfg = load_validated("configs/campus.yaml")
    assert set(cfg.hazards) == {"fall", "fire_smoke", "crowd", "intrusion"}


@pytest.mark.parametrize("mutate,needle", [
    (lambda c: c["hazards"]["fall"].__setitem__("threshold", 6), "less than 1"),
    (lambda c: c["playbooks"]["fall"][1].__setitem__("when", "groundtime > 8"), "unknown name"),
    (lambda c: c["playbooks"]["crowd"][0].__setitem__("when", "__import__('os')"), "unknown name"),
    (lambda c: c["hazards"]["crowd"]["cues"]["surge"].__setitem__("phrase", "people leaving {v:.1f}x faster"), "keyword"),
    (lambda c: c.__setitem__("cameras", [{"id": "c1", "source": "0", "zone": "Nowhere"}]), "not defined in zones"),
    (lambda c: c["hazards"]["fall"]["cues"]["torso_angle"].__setitem__("ref", 500), "must lie in"),
    (lambda c: c["runtime"].__setitem__("privcy", "raw"), "Extra inputs"),
    (lambda c: c.__setitem__("server", {"host": "0.0.0.0", "auth": {"enabled": False}}), "authentication must be enabled"),
])
def test_bad_configs_are_rejected(mutate, needle):
    c = copy.deepcopy(RAW)
    mutate(c)
    with pytest.raises(ValidationError) as e:
        Config.model_validate(c)
    assert needle in str(e.value)


# -------------------------------------------------------------------- auth
def _auth(**kw):
    users = [{"username": "asha", "role": "operator", "password_hash": hash_password("correct horse 1")},
             {"username": "vik", "role": "viewer", "password_hash": hash_password("battery staple 2")}]
    return AuthManager(users, b"x" * 48, sensor_key_digests=[key_digest("dev-key-123")], **kw)


def test_password_hashing():
    h = hash_password("a long password")
    assert verify_password("a long password", h) and not verify_password("wrong password!", h)
    with pytest.raises(ValueError):
        hash_password("short")


def test_login_roles_and_tokens():
    a = _auth()
    tok = a.login("asha", "correct horse 1")
    p = a.verify(tok)
    assert p.username == "asha" and p.can("operator") and not p.can("admin")
    assert a.verify(tok + "x") is None
    a.revoke(tok)
    assert a.verify(tok) is None
    assert a.login("asha", "nope") is None
    v = a.verify(a.login("vik", "battery staple 2"))
    assert v.can("viewer") and not v.can("operator")


def test_token_invalid_after_role_change_or_expiry():
    a = _auth(ttl_s=1)
    tok = a.login("asha", "correct horse 1")
    a.users["asha"]["role"] = "viewer"  # demoted after login
    assert a.verify(tok) is None
    a.users["asha"]["role"] = "operator"
    tok = a.issue("asha", "operator")
    time.sleep(2.1)
    assert a.verify(tok) is None


def test_bruteforce_lockout():
    a = _auth(max_failures=3, lockout_s=60)
    for _ in range(3):
        assert a.login("asha", "bad", "1.2.3.4") is None
    assert a.login("asha", "correct horse 1", "1.2.3.4") is None  # locked even with the right password
    assert a.login("asha", "correct horse 1", "5.6.7.8") is not None  # other IP unaffected


def test_device_keys():
    a = _auth()
    assert a.verify_device("dev-key-123").kind == "device"
    assert a.verify_device("dev-key-124") is None and a.verify_device(None) is None


def test_rate_limiter():
    rl = RateLimiter(per_min=60, burst=5)
    assert all(rl.allow("ip") for _ in range(5))
    assert not rl.allow("ip")
    assert rl.allow("other")


# ----------------------------------------------------------------- storage
def test_store_alerts_media_and_audit_chain(tmp_path):
    s = Store(str(tmp_path / "db.sqlite"), str(tmp_path / "media"), encrypt_media=True)
    p = s.put_media("a1_snapshot.jpg", b"\xff\xd8secret-pixels")
    assert b"secret-pixels" not in open(p, "rb").read()  # encrypted at rest
    assert s.get_media(p) == b"\xff\xd8secret-pixels"
    assert s.get_media("/etc/passwd") is None  # no path traversal
    s.save_alert({"id": "a1", "hazard": "fall", "zone": "Corridor B", "wall_time": time.time() - 40 * 86400,
                  "status": "open", "prob": 0.9, "explanation": {"text": "x"}, "snapshot_path": p})
    s.update_alert("a1", status="dismissed", operator="asha")
    assert s.get_alert("a1")["status"] == "dismissed"
    assert s.stats()["fall"]["dismissed"] == 1
    with pytest.raises(ValueError):
        s.update_alert("a1", hazard="crowd")
    for i in range(5):
        s.audit("asha", "operator", "alert_dismissed", f"a{i}", {"i": i})
    assert s.verify_audit_chain()["ok"]
    r = s.purge(30)
    assert r["incidents"] == 1 and not os.path.exists(p) and s.get_alert("a1")["subject"] == "redacted"
    # tampering with any audit row breaks the chain
    s.conn.execute("UPDATE audit SET actor = 'mallory' WHERE seq = 2")
    v = s.verify_audit_chain()
    assert not v["ok"] and v["broken_at_seq"] == 2


# ------------------------------------------------------------ notifications
class _Hook(http.server.BaseHTTPRequestHandler):
    received: list = []
    fail_first = 0

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if _Hook.fail_first > 0:
            _Hook.fail_first -= 1
            self.send_response(500)
            self.end_headers()
            return
        _Hook.received.append(json.loads(body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture()
def hook_server():
    _Hook.received, _Hook.fail_first = [], 0
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Hook)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/hook"
    srv.shutdown()


def _wait(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


ALERT = {"id": "a1", "hazard": "fire_smoke", "severity": "HIGH", "prob": 0.9, "triage": "ALARM",
         "explanation": {"headline": "Smoke / fire detected in Lab Block A", "because": "Alert raised due to x.",
                         "confidence_text": "Confidence 90% (threshold 60%)."},
         "recommendations": [{"text": "Call fire service (101)"}]}


def test_webhook_retry_and_escalation(hook_server):
    _Hook.fail_first = 1  # first delivery fails -> retried
    open_ = {"a1": True}
    n = Notifier({"ops": WebhookChannel("ops", hook_server, "slack"), "boss": WebhookChannel("boss", hook_server)},
                 [{"after_s": 0, "channels": ["ops"]}, {"after_s": 0.5, "channels": ["boss"]}],
                 is_open=lambda i: open_[i], backoff_s=0.1)
    n.alert(ALERT)
    assert _wait(lambda: len(_Hook.received) >= 2)
    assert "Smoke / fire detected" in _Hook.received[0]["text"]  # slack format
    assert _Hook.received[1]["event"] == "edgecopilot.alert"   # generic format, escalation level 1
    # acknowledged alerts do not escalate
    _Hook.received.clear()
    open_["a2"] = False
    n.alert(dict(ALERT, id="a2"))
    time.sleep(1.0)
    assert len(_Hook.received) == 1
    n.stop()


def test_webhook_rejects_plain_http():
    with pytest.raises(ValueError):
        WebhookChannel("x", "http://example.com/hook")


# --------------------------------------------------------------------- MQTT
@pytest.mark.parametrize("payload,ok", [
    (b'{"smoke_ppm": 412, "temp_c": 31.2}', True),
    (b'{"smoke_ppm": "412"}', False),
    (b'{"smoke_ppm": NaN}', False),
    (b'{"smoke_ppm": 1e9}', False),
    (b'{"temp_c": true}', False),
    (b'[1,2,3]', False),
    (b'\xff\xfe', False),
    (b'{"other": 1}', False),
    (b'{"smoke_ppm": 1}' + b' ' * 5000, False),
])
def test_mqtt_payload_validation(payload, ok):
    r = parse_sensor_payload("campus/Lab_Block_A/sensors", payload, now=100.0)
    assert (r is not None) == ok
    if ok:
        assert r["zone"] == "Lab Block A" and r["t"] == 100.0


# ------------------------------------------------------------ camera health
def test_tamper_detector():
    td = TamperDetector(frozen_s=1.0, blackout_s=1.0, defocus_s=1.0)
    rng = np.random.default_rng(0)
    t = 0.0
    for _ in range(60):  # normal, changing, textured scene
        f = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        assert not any(td(f, t).values())
        t += 0.1
    still = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    out = {}
    for _ in range(15):
        out = td(still, t)
        t += 0.1
    assert out["frozen"]
    out = {}
    for _ in range(15):
        out = td(np.full((120, 160, 3), 3, np.uint8) + rng.integers(0, 2, (120, 160, 3), dtype=np.uint8), t)
        t += 0.1
    assert out["blackout"]


def test_clip_buffer_pre_and_post():
    cb = ClipBuffer(pre_s=1.0, post_s=1.0, width=64)
    f = np.zeros((48, 64, 3), np.uint8)
    done = []
    for i in range(40):
        t = i * 0.1
        if i == 20:
            cb.mark("a1", t)
        done += cb.add(f, t)
    assert done and done[0][0] == "a1"
    frames = done[0][1]
    assert frames[0][0] <= 1.0 + 1e-6 and frames[-1][0] >= 2.9
    assert ClipBuffer.encode_mp4(frames)[4:8] in (b"ftyp", b"mdat", b"free")


def test_corroboration_rule_rejects_single_cue_fall_models():
    import pytest

    from edgecopilot.config_schema import HazardCfg
    cue = {"weight": 1.0, "ref": 0.0, "lo": 0.0, "hi": 10.0, "phrase": "x {v}", "keywords": ["x"]}
    ok = HazardCfg(bias=-12, threshold=0.6, min_threshold=0.5, min_cues=2, cues={"a": cue, "b": cue})
    assert ok.min_cues == 2
    with pytest.raises(ValueError, match="min_cues"):
        HazardCfg(bias=-8, threshold=0.6, min_threshold=0.5, min_cues=2, cues={"a": cue, "b": cue})


def test_fitted_gmdcsa_profile_is_rejected():
    import pytest

    from edgecopilot.config_schema import load_validated
    with pytest.raises(Exception, match="min_cues"):
        load_validated("benchmarks/results/experiments/fall_fitted_gmdcsa.yaml")
