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
7. Restricted areas (optional): add `restricted:` regions to the camera (polygon in 0..1 image
   coordinates, optional `hours: "20:00-07:00"` when presence is not allowed; see
   `configs/production.example.yaml`). Walk into each region once and check that an intrusion
   alert arrives after about 14 s of presence out of hours or 26 s otherwise (sooner with several people).

## Time discipline (do this on every edge box)

Alert, audit and report timestamps are stored as UTC (ISO-8601 with `Z`), and
camera kinematics use a monotonic clock, so a clock step can't corrupt fall
speeds. The *wall* clock still has to be right for evidence to hold up. IEC 62676
and CERT-In (April 2022 directions) expect clocks synchronised to an authoritative
source, and CERT-In names the NIC / NPL servers for Indian organisations:

```bash
sudo apt install chrony
sudo tee /etc/chrony/sources.d/india.sources >/dev/null <<'CONF'
server samay1.nic.in iburst
server samay2.nic.in iburst
server time.nplindia.org iburst
CONF
sudo systemctl restart chrony && chronyc tracking   # "System time" offset should be well under 1 s
```

If the campus firewall blocks NTP, use the campus NTP server instead. Check
`chronyc tracking` in the weekly checks.

## Supervision built into the service

* **Self-test** every `runtime.self_test_minutes` (default 60): for each camera it checks that
  frames are fresh, runs both models on the latest frame, and pushes a synthetic hazard through
  fusion -> explanation -> verifier -> playbook. A failure raises a `self_test_failed` trouble
  alert and the header pill turns red. An admin can run it on demand and send a clearly marked
  **TEST** message through every notification channel: `POST /api/selftest {"notify": true}`.
  Do this weekly so the people on the escalation list see that messages still arrive.
* **Long lie**: a fall alert whose person is still on the ground after `runtime.long_lie_s`
  (default 60 s) is escalated to CRITICAL / ALARM once and re-notified ("STILL DOWN").
* **Incidents**: alerts in the same zone within 2 minutes (across cameras, hazards and sensors)
  are grouped into one incident (`GET /api/incidents`), so one event is not three pages.
* **Shift handover**: the "Handover" button (operator) prints the last 12 h: open incidents
  first, outcomes, dismissals with notes, camera troubles, paused cameras, self-test state and
  the current alert lines. The export is audited.
* **What if?**: every alert card can re-score the alert with changed readings (exact, read-only),
  which is the quickest way to answer "would it still have alarmed if the sensor was wrong?".

## Daily / weekly

| When | Check | How |
|---|---|---|
| Daily | all cameras live, no tamper faults | console header / Grafana; Prometheus alerts `CameraNotLive`, `CameraTamper` |
| Daily | notifications working | Prometheus `NotificationFailures`; a weekly test alert |
| Weekly | operator precision per hazard | `GET /api/stats?hours=168` — dismissed alerts feed recalibration automatically |
| Weekly | audit log intact | `GET /api/audit/verify` (admin) must return `ok: true` |
| Weekly | clock synchronised | `chronyc tracking` offset < 1 s |
| Weekly | notification chain end to end | admin: `POST /api/selftest {"notify": true}`; every recipient confirms the TEST message |
| Each shift | handover | "Handover" button; read open incidents and degraded cameras first |
| Monthly | retention purge ran | audit entries `retention_purge` |
| Monthly | console mutes | audit entries `console_muted`; the audible alarm should normally be on at the security desk |
| Per model update | regression benchmarks | `python benchmarks/eval_*.py`; compare with docs/RESULTS.md before deploying |

## Incident response (when an alert fires)

0. With **Sound on** (header button), the console rings a two-tone alarm every 4 s while an
   ALARM-mode alert is open and unacknowledged, a single chime for VERIFY alerts, and three low
   beeps for a camera trouble signal. Muting is recorded in the audit log.
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
