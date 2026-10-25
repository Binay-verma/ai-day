"""Features ported (re-implemented) from the team's Argus project: phone push, evidence bundles,
operator track record, natural-language alert search, privacy masks, signed sensor events."""
from __future__ import annotations

import json
import urllib.request

import pytest

from edgecopilot.notify import Notifier, NtfyChannel, TelegramChannel

ALERT = {"id": "a1", "hazard": "fall", "zone": "Corridor B", "severity": "CRITICAL", "triage": "ALARM",
         "explanation": {"headline": "Fall detected (person#1) in Corridor B", "because": "Alert raised due to ...",
                         "confidence_text": "Confidence 94%"},
         "recommendations": [{"text": "Send first-aid team"}]}


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ntfy_sends_priority_tags_click_and_bearer(monkeypatch):
    seen = {}

    def fake(req, timeout=0):
        seen["url"], seen["headers"], seen["body"] = req.full_url, dict(req.header_items()), req.data
        return _Resp()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv("NTFY_TOKEN", "tk_secret")
    ch = NtfyChannel("ntfy", "https://ntfy.sh/campus-team-x9", "NTFY_TOKEN", base_url="https://copilot.lan")
    ch.send(ALERT, "text body")
    h = {k.lower(): v for k, v in seen["headers"].items()}
    assert h["priority"] == "5" and h["tags"] == "ambulance" and h["title"].startswith("Fall detected")
    assert h["click"] == "https://copilot.lan/#alert-a1" and h["authorization"] == "Bearer tk_secret"
    assert seen["body"] == b"text body"


def test_push_errors_never_leak_the_token(monkeypatch):
    def boom(req, timeout=0):
        raise OSError(f"connection refused for {req.full_url}")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setenv("TG", "123:SECRET-TOKEN")
    ch = TelegramChannel("telegram", "TG", "42")
    with pytest.raises(RuntimeError) as e:
        ch.send(ALERT, "x")
    assert "SECRET-TOKEN" not in str(e.value) and "***" in str(e.value)
    n = NtfyChannel("ntfy", "https://ntfy.sh/private-topic-123")
    with pytest.raises(RuntimeError) as e2:
        n.send(ALERT, "x")
    assert "private-topic-123" not in str(e2.value)


def test_telegram_payload_and_config_wiring(monkeypatch):
    seen = {}

    def fake(req, timeout=0):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        return _Resp()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:abc")
    n = Notifier.from_config({"telegram": {"chat_id": "77"}, "ntfy": {"url": "https://ntfy.sh/t"}},
                             is_open=lambda _id: True)
    try:
        assert set(n.channels) == {"telegram", "ntfy"}
        res = n.test("telegram")
        assert res == [{"channel": "telegram", "ok": True}]
        assert seen["url"].endswith("/bot999:abc/sendMessage") and seen["body"]["chat_id"] == "77"
        assert "test" in seen["body"]["text"].lower()
    finally:
        n.stop()


def test_ntfy_rejects_plain_http():
    with pytest.raises(ValueError):
        NtfyChannel("ntfy", "http://ntfy.example.org/topic")


# ---------------------------------------------------------------- signed evidence bundles
import io  # noqa: E402
import zipfile  # noqa: E402

from edgecopilot.evidence import build_bundle, verify_bundle  # noqa: E402
from test_server import env, login, seed_alert  # noqa: E402,F401  (pytest fixture re-used; tests/ is on sys.path)

KEY = b"k" * 48


def _bundle(**kw):
    a = dict(ALERT, snapshot_path="s.jpg", camera_id="cam1")
    media = {"s.jpg": b"\xff\xd8snapshot"}
    return build_bundle(a, key=KEY, actor="asha", get_media=media.get,
                        audit_rows=[{"seq": 2, "action": "confirmed", "target": "a1", "detail": "{\"note\": \"ok\"}"},
                                    {"seq": 1, "action": "alert_raised", "target": "a1", "detail": "{}"}],
                        audit_head="abc", report_html="<html>r</html>", **kw)


def _rezip(data, change):
    zin = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for n in zin.namelist():
            b = zin.read(n)
            b = change(n, b)
            if b is not None:
                z.writestr(n, b)
        if change("__extra__", None):
            z.writestr("extra.txt", b"planted")
    return out.getvalue()


def test_evidence_bundle_verifies_and_lists_every_file():
    res = verify_bundle(_bundle(), KEY)
    assert res["ok"], res
    assert set(res["files"]) == {"alert.json", "audit.json", "report.html", "snapshot.jpg"}
    audit = json.loads(zipfile.ZipFile(io.BytesIO(_bundle())).read("audit.json"))
    assert [e["seq"] for e in audit["entries"]] == [1, 2] and audit["entries"][1]["detail"] == {"note": "ok"}


@pytest.mark.parametrize("what, change, needle", [
    ("edited media", lambda n, b: b + b"x" if n == "snapshot.jpg" else b, "snapshot.jpg: content changed"),
    ("deleted file", lambda n, b: None if n == "report.html" else b, "report.html: listed but missing"),
    ("planted file", lambda n, b: True if n == "__extra__" else b, "extra.txt: not in the manifest"),
    ("forged manifest", lambda n, b: b.replace(b'"asha"', b'"mallory"') if n == "MANIFEST.json" else b, "signature does not match"),
])
def test_evidence_bundle_tampering_is_reported_exactly(what, change, needle):
    res = verify_bundle(_rezip(_bundle(), change), KEY)
    assert not res["ok"] and any(needle in p for p in res["problems"]), (what, res)


def test_evidence_bundle_other_site_key_fails_and_garbage_is_rejected():
    assert not verify_bundle(_bundle(), b"z" * 48)["ok"]
    assert verify_bundle(b"not a zip", KEY)["problems"] == ["not a zip file"]


def test_evidence_endpoint_exports_audits_and_verifies(env):  # noqa: F811
    c, svc = env
    seed_alert(svc)
    o = {"Authorization": f"Bearer {login(c, 'asha', 'operator pass 1')}"}
    assert c.get("/api/alerts/a1/evidence.zip", headers=o).status_code == 409  # no site secret: refuse to sign
    svc.store.audit_key = KEY
    r = c.get("/api/alerts/a1/evidence.zip", headers=o)
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert {"alert.json", "report.html", "snapshot.jpg", "audit.json", "MANIFEST.json", "SIGNATURE"} <= set(names)
    audit = json.loads(zipfile.ZipFile(io.BytesIO(r.content)).read("audit.json"))
    assert any(e["action"] == "evidence_exported" for e in audit["entries"])
    v = c.post("/api/evidence/verify", content=r.content, headers=o)
    assert v.status_code == 200 and v.json()["ok"]
    bad = c.post("/api/evidence/verify", content=_rezip(r.content, lambda n, b: b + b"!" if n == "alert.json" else b), headers=o)
    assert not bad.json()["ok"]
    vw = {"Authorization": f"Bearer {login(c, 'vik', 'viewer pass 12')}"}
    assert c.get("/api/alerts/a1/evidence.zip", headers=vw).status_code == 403
    c.cookies.clear()
    assert c.post("/api/evidence/verify", content=b"x" * (70 * 1024)).status_code == 413  # anonymous: 64 KB limit


# ---------------------------------------------------------------- operator track record
from edgecopilot.track_record import TrackRecord, wilson  # noqa: E402


def test_wilson_interval_matches_known_values():
    lo, hi = wilson(9, 11)
    assert round(lo, 3) == 0.523 and round(hi, 3) == 0.949
    assert wilson(0, 0) == (0.0, 1.0)


class _FakeStore:
    def __init__(self, rows):
        self.rows = rows

    def list_alerts(self, limit, since=None):
        return [r for r in self.rows if since is None or r["wall_time"] >= since]


def test_track_record_counts_reviews_per_camera_with_site_fallback():
    now = 1_000_000.0
    rows = ([{"hazard": "fall", "camera_id": "cam1", "status": "confirmed", "wall_time": now - 10}] * 9
            + [{"hazard": "fall", "camera_id": "cam1", "status": "dismissed", "wall_time": now - 10}] * 2
            + [{"hazard": "fall", "camera_id": "cam1", "status": "open", "wall_time": now - 10}] * 5          # not reviewed
            + [{"hazard": "fall", "camera_id": "cam1", "status": "confirmed", "wall_time": now - 40 * 86400}]  # too old
            + [{"hazard": "camera_fault", "camera_id": "cam1", "status": "dismissed", "wall_time": now}]
            + [{"hazard": "fall", "camera_id": "cam2", "status": "dismissed", "wall_time": now - 5}])
    tr = TrackRecord(_FakeStore(rows))
    r = tr.record_for("fall", "cam1", now)
    assert (r["confirmed"], r["dismissed"], r["scope"]) == (9, 2, "camera") and r["ci95"] == [0.523, 0.949]
    assert "confirmed 9 of 11" in r["text"] and "on this camera" in r["text"]
    r2 = tr.record_for("fall", "cam2", now)  # 1 review on cam2: use the site-wide record
    assert r2["scope"] == "site" and r2["reviewed"] == 12
    assert tr.record_for("fire_smoke", "cam1", now)["reviewed"] == 0


def test_alert_detail_carries_the_track_record_and_feedback_updates_it(env):  # noqa: F811
    c, svc = env
    seed_alert(svc)
    o = {"Authorization": f"Bearer {login(c, 'asha', 'operator pass 1')}"}
    assert c.get("/api/alerts/a1", headers=o).json()["track_record"]["reviewed"] == 0
    c.post("/api/alerts/a1/feedback", json={"confirmed": True}, headers=o)
    tr = c.get("/api/alerts/a1", headers=o).json()["track_record"]
    assert tr["confirmed"] == 1 and tr["scope"] == "site"
    rows = c.get("/api/track_record", headers=o).json()["rows"]
    assert any(r["hazard"] == "fall" and r["camera"] == "cam1" and r["confirmed"] == 1 for r in rows)


# ---------------------------------------------------------------- natural-language search
import datetime as dt  # noqa: E402

from edgecopilot.copilot.search import parse, run  # noqa: E402

ZONES = ["Corridor B", "Lab Block A", "Auditorium", "default"]
NOW = dt.datetime(2026, 10, 25, 14, 0).timestamp()  # Sunday 14:00 local


@pytest.mark.parametrize("q, expect", [
    ("falls in Corridor B last night", {"hazards": ["fall"], "zones": ["Corridor B"], "until_hour": 6}),
    ("how many fire alerts this week", {"hazards": ["fire_smoke"], "aggregation": "count"}),
    ("open alerts in the last 2 hours", {"statuses": ["open"], "since_ago_h": 2}),
    ("false alarms by camera yesterday", {"statuses": ["dismissed"], "aggregation": "by_camera"}),
    ("long lie in the auditorium today", {"hazards": ["fall"], "severities": ["CRITICAL"], "zones": ["Auditorium"]}),
    ("smoke or crowd alerts that need verification", {"hazards": ["fire_smoke", "crowd"], "triage": ["VERIFY"]}),
])
def test_search_parses_into_an_allow_listed_filter(q, expect):
    f = parse(q, ZONES, ["cam1", "webcam"], now=NOW)
    for k, v in expect.items():
        if k == "until_hour":
            assert dt.datetime.fromtimestamp(f["until"]).hour == v
        elif k == "since_ago_h":
            assert abs((NOW - f["since"]) / 3600 - v) < 1e-6
        else:
            assert f[k] == v, (q, k, f[k])
    assert f["understood"]  # chips are always shown


def test_search_never_invents_a_zone_and_defaults_to_24h():
    f = parse("falls in the server room", ZONES, [], now=NOW)
    assert f["zones"] == [] and f["hazards"] == ["fall"] and abs(NOW - f["since"] - 86400) < 1e-6


def test_search_runs_filters_aggregates_and_cites_ids():
    rows = [{"id": f"f{i}", "hazard": "fall", "zone": "Corridor B", "camera_id": "cam1", "severity": "HIGH",
             "status": "dismissed" if i % 2 else "confirmed", "wall_time": NOW - 3600 * i, "triage": "ALARM"} for i in range(1, 6)]
    rows += [{"id": "x1", "hazard": "fire_smoke", "zone": "Lab Block A", "camera_id": "cam2", "severity": "CRITICAL",
              "status": "open", "wall_time": NOW - 600, "triage": {"mode": "VERIFY"}}]
    st = _FakeStore(rows)
    r = run(st, parse("false alarms of falls in the last 6 hours", ZONES, ["cam1", "cam2"], now=NOW))
    assert r["total"] == 3 and {a["id"] for a in r["alerts"]} == {"f1", "f3", "f5"} and "Cited: f1" in r["summary"]
    r2 = run(st, parse("alerts by zone today", ZONES, [], now=NOW))
    assert r2["groups"][0] == {"key": "Corridor B", "count": 5}
    r3 = run(st, parse("alerts that need verification", ZONES, [], now=NOW))
    assert [a["id"] for a in r3["alerts"]] == ["x1"]


def test_search_endpoint(env):  # noqa: F811
    c, svc = env
    seed_alert(svc)
    v = {"Authorization": f"Bearer {login(c, 'vik', 'viewer pass 12')}"}
    r = c.post("/api/search", json={"q": "falls in Corridor B today"}, headers=v)
    assert r.status_code == 200 and r.json()["total"] == 1 and r.json()["alerts"][0]["id"] == "a1"
    assert c.post("/api/search", json={"q": ""}, headers=v).status_code == 422


# ---------------------------------------------------------------- privacy masks
import copy  # noqa: E402

import numpy as np  # noqa: E402

from edgecopilot.config_schema import Config, load_raw  # noqa: E402
from edgecopilot.pipeline import Copilot, load_config  # noqa: E402


def _cfg_no_models():
    c = load_config("configs/campus.yaml")
    for k in c["models"]:
        c["models"][k]["path"] = "models/__missing__.onnx"
    return c


def test_privacy_mask_blacks_out_the_polygon_before_processing():
    cop = Copilot(_cfg_no_models(), zone="Corridor B", masks=[[[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]])
    frame = np.full((240, 320, 3), 200, np.uint8)
    cop.process(frame, 0.0)
    seen = cop._last_frame
    assert seen[:, 170:].max() == 0 and seen[:, :150].min() == 200   # right half masked, left untouched
    assert frame.min() == 200                                          # caller's frame not modified


def test_privacy_mask_config_is_validated():
    c = copy.deepcopy(load_raw("configs/examples/webcam.yaml"))
    c["cameras"][0]["masks"] = [[[0.1, 0.1], [0.2, 0.1], [0.2, 0.3]]]
    Config.model_validate(c)
    c["cameras"][0]["masks"] = [[[0.1, 0.1], [1.5, 0.1], [0.2, 0.3]]]
    with pytest.raises(ValueError):
        Config.model_validate(c)
