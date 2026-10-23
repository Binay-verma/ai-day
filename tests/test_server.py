from fastapi.testclient import TestClient

from edgecopilot.copilot.events import Alert, EventLog
from edgecopilot.pipeline import Copilot, load_config
from edgecopilot.server.app import create_app


def make_client(tmp_path):
    cfg = load_config("configs/campus.yaml")
    for k in cfg["models"]:
        cfg["models"][k]["path"] = "models/__missing__.onnx"
    cop = Copilot(cfg, root=str(tmp_path), zone="Corridor B", event_log=EventLog())
    return cop, TestClient(create_app(cop, None))


def test_api_roundtrip(tmp_path):
    cop, c = make_client(tmp_path)
    assert c.get("/").status_code == 200
    assert c.get("/api/status").json()["zone"] == "Corridor B"
    al = Alert(hazard="fall", subject="person#1", zone="Corridor B", t_video=1.0, prob=0.9, severity="HIGH",
               explanation={"text": "x", "headline": "Fall", "because": "b", "counterfactual": "", "confidence_text": "Confidence 90% (threshold 60%)."},
               decision={"counterfactuals": [], "sufficient_reason": ["torso_angle"], "margin_logit": 1.8},
               verification={"groundedness": 1.0, "n_claims": 3, "unsupported": []},
               recommendations=[{"text": "Send first aid", "priority": 1, "because": "standard procedure"}],
               cues=[{"name": "torso_angle", "value": 80, "unit": "deg", "contribution": 3.0, "reliability": 1.0,
                      "phrase": "torso tilted 80 deg", "source": "vision"}])
    cop.events.add(al)
    assert c.get("/api/alerts").json()[0]["id"] == al.id
    assert "snapshot_jpg_b64" in c.get(f"/api/alerts/{al.id}").json()
    r = c.post(f"/api/alerts/{al.id}/ask", json={"question": "why?"}).json()
    assert "torso" in r["answer"]
    r = c.post(f"/api/alerts/{al.id}/feedback", json={"confirmed": False}).json()
    assert r["bias_offset"] < 0
    assert c.post("/api/sensors", json={"zone": "Corridor B", "t": 1.0, "smoke_ppm": 400}).json()["ok"]
    assert c.get("/api/alerts/nope").status_code == 404
