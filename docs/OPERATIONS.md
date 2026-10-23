# Operations runbook

## Install (edge box: Raspberry Pi 5 8 GB / Jetson Orin Nano / any x86 mini-PC)

```bash
git clone <repo> /opt/edgecopilot && cd /opt/edgecopilot
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[mqtt]"                       # runtime only: onnxruntime, opencv, fastapi (no PyTorch)
cp .env.example .env && $EDITOR .env           # secrets
python -m edgecopilot.security hash            # -> paste into server.auth.users in configs/campus.yaml
python -m edgecopilot.security keygen          # one per ESP32 sensor node
# copy models/*.onnx (from the release or `python scripts/setup_models.py` on a dev machine)
python -m edgecopilot.config_schema configs/campus.yaml   # must print OK
sudo cp deploy/systemd/edgecopilot.service /etc/systemd/system/ && sudo systemctl enable --now edgecopilot
```

Docker alternative: `docker compose up -d` (copilot + MQTT broker + Prometheus + Grafana).

## Commissioning a camera (do this once per camera)

1. Add it to `cameras:` in the config (id, `rtsp://…` source, zone, hazards to enable).
2. Mount/aim: whole floor area visible, people ≥ 80 px tall, avoid pointing at windows/sun.
3. Record **normal** footage at different times of day (≥ 2 h in total is a good start).
4. `python scripts/calibrate_zone.py --zone "<zone>" --video normal1.mp4 normal2.mp4 --alpha 0.01`
   — sets that camera's alert line to a stated false-alarm budget.
5. Run a supervised test: walk-through, simulated fall onto a mat, smoke pen / theatrical smoke
   **with the fire officer's permission**; check that alerts arrive within seconds and notifications reach the right people.
6. Record the commissioning result in the safety log (date, tester, alerts observed).

## Daily / weekly

| When | Check | How |
|---|---|---|
| Daily | all cameras live, no tamper faults | console header / Grafana; Prometheus alerts `CameraNotLive`, `CameraTamper` |
| Daily | notifications working | Prometheus `NotificationFailures`; a weekly test alert |
| Weekly | operator precision per hazard | `GET /api/stats?hours=168` — dismissed alerts feed recalibration automatically |
| Weekly | audit log intact | `GET /api/audit/verify` (admin) must return `ok: true` |
| Monthly | retention purge ran | audit entries `retention_purge` |
| Per model update | regression benchmarks | `python benchmarks/eval_*.py`; compare with docs/RESULTS.md before deploying |

## Incident response (when an alert fires)

1. Read the explanation and the evidence ledger; **VERIFY** means look first, **ALARM** means act.
2. Follow the recommended actions (they cite their trigger and the zone's AED / exits / wardens).
3. Press **Confirm** or **Dismiss** — this is recorded with your name and tunes that camera.
4. For confirmed incidents, export the **Incident report** for the safety committee.

## Failure modes and what to do

| Symptom | Likely cause | Action |
|---|---|---|
| Camera `offline`, reconnect count rising | network / camera power / RTSP credentials | check PoE switch and camera web UI; the worker retries with back-off up to every 30 s |
| `frozen` fault | encoder stuck, someone looped a still image | reboot camera; investigate tampering |
| `blackout` fault | lens covered, IR failure at night | inspect physically |
| `defocus` fault | lens sprayed / moved | inspect, re-aim, re-calibrate |
| Many dismissed alerts on one camera | scene unlike calibration footage (new furniture, lighting) | record new normal footage and re-run `calibrate_zone.py` |
| Latency p95 > 400 ms | too many cameras per box | use the 416 px or INT8 models, raise `fire_every`, or add an edge box |

## Limits that operators must know

* This is a **decision-support** tool. It is **not a certified fire alarm** (IS 2189 / EN 54 / UL 268 detectors and the building's fire alarm panel remain the life-safety system) and not a medical device.
* Falls onto beds and partially visible people are the weakest case (see docs/RESULTS.md §2).
* A person lying down deliberately can raise a VERIFY fall alert.
