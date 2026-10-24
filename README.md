# Explainable Edge AI Copilot for Smart Emergency Response

**Team local maxima** — Binay Verma, Chetan Jagannath, Bhyresh B S, Dhanush D

An on-device copilot for campus safety operators. It watches camera feeds and
IoT sensors on a CPU-only edge box, detects **falls, smoke/fire and abnormal
crowd behaviour**, and for every alert tells the operator — in plain language
that is *checked against the evidence* — why it fired, what would have changed
the decision, how sure it is, and what to do next.

```
FALL — CRITICAL — ALARM
Fall detected (person#1) in Corridor B. Alert raised due to a horizontal body shape
(width/height 3.1), torso tilted 81 deg from vertical and the torso rotating 152 deg/s.
It would not have fired if tilt rate had been below 51.46 deg/s (observed 151.70 deg/s).
Confidence 91% (threshold 60%).
Recommended: Send first-aid team (ext. 2311) to Corridor B for person#1; ...
Explanation checked against the evidence: 100% of 14 claims supported · decided in 137 ms on-device
```

## Why this is not "just another detector"

Survey of ≈95 products and papers: [`docs/research/landscape.md`](docs/research/landscape.md).
Commercial systems (Verkada, Milestone, Ambient.ai, NVIDIA VSS) and research
systems (Holmes-VAD, VAD-LLaMA, LAVAD) produce text explanations with large
GPU/cloud models and never check that text against what the detector measured.
What this project adds:

| Add-on | What it gives the operator |
|---|---|
| **Evidence ledger** (additive log-odds fusion of named, unit-bearing cues) | The explanation *is* the decision: per-cue contributions sum exactly to the score (exactness error ≈ 1e-15) |
| **Grounding verifier** | Every number, cause and hazard in the text is checked against the ledger; LLM-style errors are caught before the operator sees them |
| **Exact counterfactuals + sufficient reasons** | "It would not have fired if …" — computed in closed form, 100 % valid |
| **Conformal false-alarm budget per camera** | Thresholds calibrated on that camera's normal footage with a stated α; dismissals feed back into calibration |
| **ALARM vs VERIFY triage** | Conflicting evidence (e.g. camera sees smoke, MQ-2 reads clean air) is routed to a human check instead of an evacuation |
| **Cross-hazard corroboration** | Rapid exit / dispersal of people raises fire risk and vice-versa |
| **Evidence-cited SOP recommendations** | Campus playbook (exits, AED, extinguisher, 112/108/101) instantiated per zone; each step cites its trigger |
| **Ask the copilot** | Offline, grounded Q&A over the alert ("what if the sensor is wrong?") |
| **Privacy by design** | Heads blurred using pose key-points (no face detector), skeleton-only mode, no raw video leaves the box |
| **Edge engineering** | ONNX Runtime, motion-gated cascade, scene-change/tamper reset, re-alert cooldown, Prometheus metrics, Docker |

## Architecture

```
 camera ──► motion gate ──► YOLO11n-pose (ONNX) ──► tracker ──► fall cues ──────────────┐
    │                  └──► fire/smoke YOLO11n (ONNX, + neck feature for Eigen-CAM) ─────┤
    │                       dark-channel smoke density · flame flicker ──────────────────┤──► evidence ledger
    └──► Farneback optical flow ──► surge · exodus · divergence · coherence ─────────────┤    (per hazard)
 ESP32 (MQ-2 ppm, temp rise, sound dB) ──► sensor hub with staleness ───────────────────┘          │
                                                                                                    ▼
   hysteresis ─► conformal alert line ─► triage ─► counterfactuals ─► template text ─► verifier ─► SOP actions
                                                  evidence heat-map (Eigen-CAM / skeleton / flow) ─┘
                                                                    ▼
                     FastAPI + WebSocket operator console  ◄──► operator feedback ─► recalibration
```

Code map:

| Path | What |
|---|---|
| `edgecopilot/perception/` | ONNX YOLO runtime, tracker, fall / crowd / smoke cue extractors |
| `edgecopilot/xai/` | fusion (ledger), counterfactuals, NLG + verifier, triage, calibration, saliency |
| `edgecopilot/copilot/` | recommender (safe playbook conditions), events, Q&A |
| `edgecopilot/pipeline.py` | per-frame loop: gate → perception → fusion → explanation → action |
| `edgecopilot/server/` | FastAPI app + operator console (`static/index.html`) |
| `configs/campus.yaml` | **everything auditable**: cue weights, thresholds, zones, playbooks |
| `benchmarks/` | reproducible evaluations (results in `benchmarks/results/`) |
| `hardware/esp32_sensor_node/` | Arduino sketch for the MQ-2 / DHT22 / sound sensor node |

## Results

All measured on a **2-core CPU, no GPU**, with 95 % intervals. Full protocols, caveats and
failure analysis: [`docs/RESULTS.md`](docs/RESULTS.md). The fall, crowd and fire-video sets
were also used during development, so those numbers are optimistic; the leakage-free fire
image benchmark and the frozen-system external check are not.

| Area | Result |
|---|---|
| Explanation exactness | per-cue contributions sum to the score with max error **1.6e-15** (73 real alerts) |
| Grounded text | **0 %** unsupported claims; verifier catches **99.8 %** of injected narrator errors (8 types) at **0.5 %** false rejects |
| Counterfactuals | **100 %** valid (318); explanation + verification ≈ 1 ms |
| Falls (GMDCSA24, leave-one-subject-out) | F1 **0.81–0.83** default, **0.84** with the YOLO11s-pose profile. Target 0.90: not met |
| Falls, second dataset never used in development (UR-Fall) | **23/30** with the YOLO11s-pose profile (19/30 default), 0/5 alarms on daily activities |
| Abnormal crowd movement (UMN) | **10/11** dispersal events, 0 false alarms in 200 s of normal video, with per-camera conformal calibration |
| Fire/smoke detector (leakage-free split) | mAP@0.5 **0.56** Home-fire / **0.65** DBA-Fire scenes (start checkpoint 0.37 / 0.36); 15.5 % false positives on neutral images, so it is one cue among several, never the alarm on its own |
| Fire, system level (FURG videos) | **15/17** fire videos detected (median 2.7 s), **0** false alarms in 4.7 min of non-fire video incl. moving cameras |
| External check (frozen system, unseen footage) | **9/11** falls with the fall profile (6/11 default); 1 false alarm in 7.2 min (crowd, uncalibrated camera), 0 false fall alarms |
| Privacy filter, verified | an independent face detector finds **0 of 184** faces after the filter |
| Edge speed (all 3 hazards at once) | **12.0 FPS** with the 416 px production models; 7.8 INT8-640; 6.6 FP32-640 (the audit's safety fixes cost ~40 % at 640 px) |
| Saliency honesty check | Eigen-CAM beats random but not a centre-bias control, so the ledger, not the heat map, is the explanation |

## Production features

| Area | What is in the box |
|---|---|
| Access control | FastAPI with RBAC (viewer < operator < admin), scrypt passwords, pinned HS256 JWT in HttpOnly cookie, revocation, lockout, rate and body limits, CSP, WebSocket origin check |
| Devices | per-device API keys bound to zones; MQTT over TLS where the topic, not the payload, decides the zone |
| Evidence and privacy | SQLite (WAL) incident store, HMAC-keyed hash-chained audit log, Fernet-encrypted media, retention purge, legal hold, erasure; faces pixelated before anything is stored or streamed, **verified with an independent face detector (0 of 184 faces still detectable)** |
| Who watches the watcher | trouble alerts for offline / frozen / blackout / defocus / view-changed / processing-error cameras through the same escalation chain; systemd watchdog; `/healthz` fails on a dead camera thread; audible console alarm with distinct alarm, verify and trouble tones (muting is audited); **hourly end-to-end self-test** (frames, both models, fusion → explanation → verifier → playbook) with a TEST-message option |
| Human control | ALARM vs VERIFY triage, confirm / dismiss once per alert, operator pause / resume per camera (audited), calibrated alert lines clipped to `[min_threshold, max_threshold]` so dismissals can never silence a hazard; **What-if explorer** (drag a reading, see the exact new score) |
| Emergency workflow | **long-lie escalation** (still down after 60 s → CRITICAL + re-notify), **cross-camera incident grouping** (one incident, not three pages), **shift-handover report**, **restricted-area intrusion** rules with forbidden hours |
| Notifications | webhook, Slack, Teams, email, MQTT with retries and escalation; every fire message carries the decision-support disclaimer |
| Operations | multi-camera workers with reconnect, frame dropping and clip buffer; Prometheus metrics, Alertmanager, SLO burn-rate and model-drift rules (dismissal rate, alert line at ceiling); Docker compose with Caddy TLS, Mosquitto, Prometheus, Grafana |
| Supply chain | strict pydantic config schema with `extends:` overlays; SHA-256-pinned models (refused on mismatch); CycloneDX SBOM + ML-BOM; lockfile; CI with ruff, mypy, bandit, pip-audit, tests and SLSA provenance |

The independent audits (20 code defects, 13 security findings, evaluation leakage) and what
was fixed are summarised in [`docs/AUDIT.md`](docs/AUDIT.md), together with an honest
production-readiness verdict.

> **Decision support only.** This system supplements, and never replaces, the certified fire
> alarm system, trained first responders and emergency services (112 / 108 / 101). It never
> dispatches anyone automatically; every recommendation is for a human to act on.

## Quick start

```bash
pip install -e ".[dev]"            # runtime: onnxruntime + opencv + fastapi (no torch)
# models + demo video: unzip the release asset edge-copilot-models-and-demo.zip into the repo root
#   (or build them yourself: pip install -e ".[train]" && python scripts/setup_models.py,
#    and EDGECOPILOT_DATASETS=<datasets folder> python scripts/make_demo.py)
python -m edgecopilot.server.app --source demo/demo_campus.mp4 --demo   # open http://127.0.0.1:8000
python -m edgecopilot.cli --source 0          # headless, webcam
pytest -q
```

Production stack (TLS, MQTT, monitoring): copy `.env.example` to `.env`, then `docker compose up -d`; see `docs/OPERATIONS.md`.

Fall-accurate profile (YOLO11s-pose, ~3x CPU): `--config configs/profiles/fall_accurate.yaml`, or per camera with `cameras[].pose` (see `configs/production.example.yaml`).

RTSP camera: `--source rtsp://user:pass@camera/stream`. ESP32 node posts to `/api/sensors`.

## Documentation

| Doc | What |
|---|---|
| [docs/AUDIT.md](docs/AUDIT.md) | audit findings, fixes, standards position, verdict |
| [docs/RESULTS.md](docs/RESULTS.md) | every benchmark, with intervals and caveats |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | components and data flow |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | install, commissioning, time sync, incident response, failure modes |
| [docs/SECURITY.md](docs/SECURITY.md) · [docs/PRIVACY.md](docs/PRIVACY.md) | threat model, hardening, DPDP-oriented privacy notes |
| [docs/RISK_REGISTER.md](docs/RISK_REGISTER.md) · [docs/AI_GOVERNANCE.md](docs/AI_GOVERNANCE.md) · [docs/SLO.md](docs/SLO.md) | FMEA, ISO 42001 / NIST AI RMF mapping, service levels |
| [models/README.md](models/README.md) | model cards |
| [docs/research/](docs/research/) | landscape survey, methods, standards gap analysis, audit reports |

## Licences and data

Code: MIT. Datasets are used for evaluation only and are not redistributed:
GMDCSA24 (MIT), UMN crowd (research use), FURG fire (CC0), Home-fire (CC BY-NC 4.0),
DBA-Fire, DeepQuest fire/smoke, PETS-2009 `vtest`. Ultralytics weights are AGPL-3.0;
see `models/README.md`.
