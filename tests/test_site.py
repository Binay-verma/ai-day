"""Site awareness: safe exits, crowd heading into danger, went-down-and-left-the-view, heading prediction."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from test_server import env, login  # noqa: F401  (pytest fixture re-used; tests/ is on sys.path)

from edgecopilot.copilot.recommender import Recommender
from edgecopilot.perception.detector import Detection, DetectorOutput
from edgecopilot.perception.exits import ExitMonitor, Heading, flow_summary, predict_edge
from edgecopilot.perception.tracker import Track, TrackState
from edgecopilot.pipeline import Copilot, load_config
from edgecopilot.site import SiteState
from edgecopilot.xai.fusion import HazardModel
from edgecopilot.xai.nlg import explain, numbers_in, verify


def _cfg():
    c = load_config("configs/campus.yaml")
    for k in c["models"]:
        c["models"][k]["path"] = "models/__missing__.onnx"
    c["site"] = {"exits": {"East stairwell": {"cameras": ["cam2"]}, "West stairwell": {"cameras": ["cam3"]}},
                 "hazard_ttl_s": 60, "crowd_toward_hazard": {"min_people": 3, "cooldown_s": 60, "max_eta_s": 20}}
    return c


FIRE = {"fire_smoke": {"prob": 0.78, "threshold": 0.6, "active": True}}
CALM = {"fire_smoke": {"prob": 0.05, "threshold": 0.6, "active": False}}


def _site(c=None):
    return SiteState(c or _cfg(), {"cam1": "Corridor B", "cam2": "Corridor B", "cam3": "Corridor B"})


# ------------------------------------------------------------------ 1. safe exits
def test_smoke_at_an_exit_blocks_it_and_routes_to_the_other():
    s = _site()
    tr = s.observe("cam2", FIRE, now=100.0)
    assert [t["type"] for t in tr] == ["exit_blocked"] and tr[0]["exit"] == "East stairwell"
    use, avoid = s.route_advice("Corridor B", now=101.0)
    assert use == ["West stairwell"] and avoid[0][0] == "East stairwell"
    assert "camera cam2 (78%, alert line 60%)" in avoid[0][1]
    assert s.observe("cam2", FIRE, now=102.0) == []                      # raised once, not every frame
    st = s.exit_status(now=102.0)
    assert st["East stairwell"]["state"] == "unsafe" and st["West stairwell"]["state"] == "clear"


def test_exit_stays_unsafe_for_the_ttl_then_clears():
    s = _site()
    s.observe("cam2", FIRE, now=100.0)
    assert s.observe("cam2", CALM, now=150.0) == []                      # within 60 s: still unsafe
    assert [t["type"] for t in s.observe("cam2", CALM, now=161.0)] == ["exit_clear"]
    assert s.route_advice("Corridor B", now=161.0) == (["East stairwell", "West stairwell"], [])


def _fire_assessment(c, prob_cue=4.0):
    m = HazardModel.from_config("fire_smoke", c["hazards"]["fire_smoke"])
    a = m.assess("Corridor B", 0.0, {"fire_conf": 0.8, "smoke_conf": 0.7, "smoke_density": 0.5, "smoke_ppm": 900})
    return m, a


def test_recommendations_route_to_safe_exits_and_say_why_and_verify():
    c = _cfg()
    s = _site(c)
    s.observe("cam2", FIRE, now=1e12)  # far future so the default clock (time.time) is inside the TTL
    r = Recommender.from_config(c)
    r.route_fn = lambda z: s.route_advice(z, now=1e12)
    m, a = _fire_assessment(c)
    texts = [x["text"] for x in r.recommend(a, "Corridor B")["actions"]]
    evac = next(t for t in texts if t.startswith("Trigger evacuation"))
    assert "via West stairwell;" in evac and "East" not in evac
    assert any(t.startswith("Do NOT send people to East stairwell: smoke / fire seen by camera cam2") for t in texts)
    e = explain(a, m, "Corridor B", texts[:3])
    v = verify(e.text, a, m, "Corridor B", extra_numbers=numbers_in(texts[:3]), recommendations=texts[:3])
    assert v.groundedness == 1.0, v.unsupported


def test_no_safe_exit_is_said_plainly():
    c = _cfg()
    s = _site(c)
    s.observe("cam2", FIRE, now=1e12)
    s.observe("cam3", FIRE, now=1e12)
    r = Recommender.from_config(c)
    r.route_fn = lambda z: s.route_advice(z, now=1e12)
    _, a = _fire_assessment(c)
    texts = [x["text"] for x in r.recommend(a, "Corridor B")["actions"]]
    assert any("every listed exit reports a hazard" in t for t in texts)
    assert any(t.startswith("Every listed exit of Corridor B reports a hazard") for t in texts)
    assert sum(t.startswith("Do NOT send people to") for t in texts) == 2


def test_fall_recommendations_do_not_mention_exits():
    c = _cfg()
    s = _site(c)
    s.observe("cam2", FIRE, now=1e12)
    r = Recommender.from_config(c)
    r.route_fn = lambda z: s.route_advice(z, now=1e12)
    m = HazardModel.from_config("fall", c["hazards"]["fall"])
    a = m.assess("person#1", 0.0, {"torso_angle": 90, "aspect_ratio": 3, "height_drop": 0.7, "tilt_rate": 150,
                                   "descent_speed": 0.9, "hip_drop": 0.45, "ground_time": 1.0})
    assert not any("Do NOT" in x["text"] for x in r.recommend(a, "Corridor B")["actions"])


# ------------------------------------------------------------------ 2. crowd heading into danger
def test_crowd_heading_to_an_unsafe_exit_raises_once_with_the_alternative():
    s = _site()
    s.observe("cam2", FIRE, now=100.0)
    flows = {"East stairwell": {"people": 4, "eta_median_s": 5.0}}
    out = s.crowd_toward_hazard("cam1", flows, now=101.0)
    assert len(out) == 1 and out[0]["place"] == "East stairwell" and out[0]["alternatives"] == ["West stairwell"]
    assert s.crowd_toward_hazard("cam1", flows, now=110.0) == []                       # cooldown
    assert s.crowd_toward_hazard("cam1", {"East stairwell": {"people": 2, "eta_median_s": 5}}, now=200.0) == []
    assert s.crowd_toward_hazard("cam1", {"East stairwell": {"people": 5, "eta_median_s": 45}}, now=200.0) == []
    assert s.crowd_toward_hazard("cam1", {"West stairwell": {"people": 5, "eta_median_s": 5}}, now=200.0) == []


def test_crowd_heading_to_a_zone_with_fire():
    c = _cfg()
    s = SiteState(c, {"cam1": "Corridor B", "lab": "Lab Block A"})
    s.observe("lab", FIRE, now=10.0)
    out = s.crowd_toward_hazard("cam1", {"Lab Block A": {"people": 6, "eta_median_s": 3.0}}, now=11.0)
    assert out and out[0]["evidence"].camera == "lab"


def test_service_raises_site_alerts_end_to_end(env):  # noqa: F811
    c, svc = env
    svc.site = _site()
    svc.site.camera_zones.update({"cam1": "Corridor B", "cam2": "Corridor B"})
    svc._site_update("cam2", {"gauges": FIRE})
    svc._site_update("cam1", {"gauges": CALM, "flows": {"East stairwell": {"people": 5, "eta_median_s": 4.0}}})
    rows = {r["hazard"]: r for r in svc.store.list_alerts(10)}
    eb, ct = rows["exit_blocked"], rows["crowd_toward_hazard"]
    assert eb["subject"] == "East stairwell" and "Redirect evacuation of Corridor B to West stairwell" in \
        [x["text"] for x in eb["recommendations"]]
    assert "camera cam2 (78%, alert line 60%)" in eb["explanation"]["because"]
    assert ct["severity"] == "CRITICAL" and "to West stairwell" in ct["recommendations"][0]["text"]
    o = {"Authorization": f"Bearer {login(c, 'asha', 'operator pass 1')}"}
    st = c.get("/api/site", headers=o).json()
    assert st["exits"]["East stairwell"]["state"] == "unsafe"


# ------------------------------------------------------------------ heading prediction
def test_predict_edge_uses_direction_when_moving_and_position_when_slow():
    box = np.array([300, 200, 340, 300], float)        # 100 px tall, centre (320, 250) of a 640x480 image
    assert predict_edge(box, np.array([120.0, 0.0]), (480, 640))[0] == "right"
    assert predict_edge(box, np.array([-120.0, 10.0]), (480, 640))[0] == "left"
    near_left = np.array([20, 200, 60, 300], float)
    assert predict_edge(near_left, np.array([5.0, 0.0]), (480, 640))[0] == "left"  # barely moving: where they are
    edge, eta, _ = predict_edge(box, np.array([100.0, 0.0]), (480, 640))
    assert abs(eta - 3.2) < 1e-6


def test_flow_summary_groups_by_place():
    hs = [Heading(1, "right", 2.0, 1.0), Heading(2, "right", 4.0, 1.0), Heading(3, "left", 1.0, 1.0)]
    f = flow_summary(hs, {"right": "East stairwell"})
    assert f == {"East stairwell": {"people": 2, "edge": "right", "eta_median_s": 3.0}}


def test_exit_monitor_classifies_edge_exits_and_vanishing():
    mon = ExitMonitor()

    def track(i, boxes):
        tr = Track(i)
        for k, b in enumerate(boxes):
            tr.history.append(TrackState(k / 10, np.array(b, float), None, 0.9))
        return tr
    a = track(1, [[500 + 20 * k, 200, 540 + 20 * k, 300] for k in range(5)])  # walking right, reaches x2=620
    b = track(2, [[300, 200, 340, 300]] * 5)                                   # standing mid-image
    mon.update({1: a, 2: b}, (480, 640))
    ev = {e.track_id: e for e in mon.update({}, (480, 640))}
    assert ev[1].edge == "right" and ev[2].edge is None


# ------------------------------------------------------------------ 3. went down and left the view
class _Replay:
    def __init__(self):
        self.dets = []

    def __call__(self, frame):
        return DetectorOutput(list(self.dets), None, (1, 0, 0, 0, 0), 0.0)


def _kp(box, lying=False):
    x1, y1, x2, y2 = box
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    if lying:  # shoulders left, hips right: horizontal torso
        k[5, :2], k[6, :2] = (x1 + 10, y1 + 10), (x1 + 10, y2 - 10)
        k[11, :2], k[12, :2] = (x1 + 0.6 * (x2 - x1), y1 + 10), (x1 + 0.6 * (x2 - x1), y2 - 10)
    else:
        cx = (x1 + x2) / 2
        k[5, :2], k[6, :2] = (cx - 10, y1 + 0.2 * (y2 - y1)), (cx + 10, y1 + 0.2 * (y2 - y1))
        k[11, :2], k[12, :2] = (cx - 8, y1 + 0.55 * (y2 - y1)), (cx + 8, y1 + 0.55 * (y2 - y1))
    return k


def _run_copilot(seq, edges=None):
    c = _cfg()
    c["runtime"]["motion_gate"] = False
    rp = _Replay()
    cop = Copilot(c, zone="Corridor B", pose_model=rp, enable={"fall": True, "crowd": False, "fire_smoke": False},
                  edges=edges)
    frame = np.zeros((480, 640, 3), np.uint8)
    alerts = []
    for k, (box, lying) in enumerate(seq):
        rp.dets = [] if box is None else [Detection(np.array(box, float), 0.9, 0, "person", _kp(box, lying))]
        alerts += cop.process(frame, k / 15).new_alerts
    return alerts


STAND = [(300, 120, 360, 420)] * 20
DROP = [(300 - 10 * k, 120 + 40 * k, 360 + 25 * k, 420 + 8 * k) for k in range(1, 7)]


def test_fall_that_stays_in_view_is_an_ordinary_alarm_without_a_duplicate():
    seq = [(b, False) for b in STAND] + [(tuple(min(v, 479) for v in b), k >= 3) for k, b in enumerate(DROP)]
    seq += [((150, 400, 560, 479), True)] * 2 + [(None, False)] * 40
    alerts = _run_copilot(seq, edges={"bottom": "Lab Block A"})
    assert [a.explanation["headline"] for a in alerts] == ["Fall detected (person#1) in Corridor B"]


def test_person_dropping_out_of_view_raises_a_verify_alert_naming_the_place():
    seq = [(b, False) for b in STAND] + [(tuple(min(v, 479) for v in b), k >= 4) for k, b in enumerate(DROP[:5])]
    seq += [(None, False)] * 40
    alerts = _run_copilot(seq, edges={"bottom": "Lab Block A"})
    assert len(alerts) == 1
    a = alerts[0]
    assert a.explanation["headline"] == "Possible fall out of view (person#1) in Corridor B"
    assert a.triage["mode"] == "VERIFY" and a.severity == "HIGH"
    assert a.recommendations[0]["text"].startswith("Check Lab Block A now")
    assert any(c["name"] == "left_view" and c["contribution"] == 0.0 for c in a.cues)  # ledger unchanged
    assert a.verification["groundedness"] == 1.0
    no_map = _run_copilot(seq)  # without an edge map it still says where to look
    assert "beyond the bottom edge" in no_map[0].recommendations[0]["text"]


def test_walking_out_of_view_upright_raises_nothing():
    seq = [((100 + 15 * k, 120, 160 + 15 * k, 420), False) for k in range(35)] + [(None, False)] * 40
    assert _run_copilot(seq, edges={"right": "East stairwell"}) == []


def test_site_config_is_validated():
    from edgecopilot.config_schema import Config, load_raw
    c = copy.deepcopy(load_raw("configs/examples/webcam.yaml"))
    c["site"] = {"exits": {"East stairwell": {"cameras": ["webcam"]}}}
    c["cameras"][0]["edges"] = {"left": "East stairwell", "right": "Lab Block A"}
    Config.model_validate(c)
    c["site"]["exits"]["Back door"] = {}
    with pytest.raises(ValueError, match="not listed in any zone"):
        Config.model_validate(c)
    del c["site"]["exits"]["Back door"]
    c["cameras"][0]["edges"]["top"] = "Narnia"
    with pytest.raises(ValueError, match="neither an exit nor a zone"):
        Config.model_validate(c)


def test_site_alerts_can_be_reviewed_asked_and_exported(env):  # noqa: F811
    c, svc = env
    svc.site = _site()
    svc.site.camera_zones.update({"cam2": "Corridor B"})
    svc._site_update("cam2", {"gauges": FIRE})
    aid = svc.store.list_alerts(5)[0]["id"]
    o = {"Authorization": f"Bearer {login(c, 'asha', 'operator pass 1')}"}
    assert c.get(f"/api/alerts/{aid}", headers=o).status_code == 200
    assert c.post(f"/api/alerts/{aid}/ask", json={"question": "Why?"}, headers=o).status_code == 200
    assert c.get(f"/api/alerts/{aid}/report.html", headers=o).status_code == 200
    r = c.post(f"/api/alerts/{aid}/feedback", json={"confirmed": True}, headers=o)
    assert r.status_code == 200 and r.json()["status"] == "confirmed"
    hits = c.post("/api/search", json={"q": "exit alerts today"}, headers=o).json()
    assert hits["total"] >= 1


def test_demo_site_scenario_drives_the_real_logic_and_is_labelled(env):  # noqa: F811
    c, svc = env
    o = {"Authorization": f"Bearer {login(c, 'root', 'admin pass 123')}"}
    assert c.post("/api/demo/site_scenario", headers=o).status_code == 404  # never with authentication on
    r = svc.site_scenario()
    assert r["blocked_exit"] == "East stairwell" and r["exits"]["East stairwell"]["state"] == "unsafe"
    heads = sorted(a["explanation"]["headline"] for a in svc.store.list_alerts(10))
    assert len(heads) == 2 and all(h.startswith("DEMO SCENARIO - ") for h in heads)


def test_scene_change_or_frame_gap_is_not_a_person_leaving():
    mon = ExitMonitor()
    tr = Track(1)
    for k in range(5):
        tr.history.append(TrackState(k / 10, np.array([200, 380, 600, 479], float), None, 0.9))  # lying at the bottom edge
    mon.update({1: tr}, (480, 640), {1: {"low": True, "prob": 0.5, "cues": []}})
    mon.reset()
    assert mon.update({}, (480, 640)) == []
    # in the pipeline: a person going down at the bottom edge, then the picture changes (scene cut)
    seq = [(b, False) for b in STAND] + [(tuple(min(v, 479) for v in b), k >= 4) for k, b in enumerate(DROP[:5])]
    c = _cfg()
    c["runtime"]["motion_gate"] = False
    rp = _Replay()
    cop = Copilot(c, zone="Corridor B", pose_model=rp, enable={"fall": True, "crowd": False, "fire_smoke": False})
    frame = np.zeros((480, 640, 3), np.uint8)
    alerts = []
    for k, (box, lying) in enumerate(seq):
        rp.dets = [Detection(np.array(box, float), 0.9, 0, "person", _kp(box, lying))]
        alerts += cop.process(frame, k / 15).new_alerts
    rp.dets = []
    t = len(seq) / 15
    for j in range(1, 40):  # frames resume after a 5 s gap: the old track must not count as having left
        alerts += cop.process(frame, t + 5 + j / 15).new_alerts
    assert not [a for a in alerts if a.explanation["headline"].startswith("Possible fall out of view")]


def test_flow_eta_is_json_safe_and_not_approaching_never_alerts():
    import json
    f = flow_summary([Heading(1, "right", float("inf"), 0.5), Heading(2, "right", float("inf"), 0.5)], {"right": "East stairwell"})
    assert f["East stairwell"]["eta_median_s"] is None
    json.loads(json.dumps(f, allow_nan=False))  # browsers reject Infinity in JSON
    s = _site()
    s.observe("cam2", FIRE, now=100.0)
    assert s.crowd_toward_hazard("cam1", {"East stairwell": {"people": 5, "eta_median_s": None}}, now=101.0) == []
    assert s.crowd_toward_hazard("cam1", {"East stairwell": {"people": 5, "eta_median_s": 4.0}}, now=102.0)
