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

See [`docs/RESULTS.md`](docs/RESULTS.md) for full numbers, protocols and failure analysis.

## Quick start

```bash
pip install -e ".[dev]"                       # runtime: onnxruntime + opencv + fastapi (no torch)
python scripts/setup_models.py                # download + export models to ONNX (needs the [train] extra once)
python scripts/make_demo.py                   # builds demo/demo_campus.mp4 from public datasets
python -m edgecopilot.server.app --source demo/demo_campus.mp4   # open http://127.0.0.1:8000
python -m edgecopilot.cli --source 0          # headless, webcam
pytest -q
```

Docker (CPU-only edge image): `docker build -t edge-copilot . && docker run -p 8000:8000 edge-copilot`

RTSP camera: `--source rtsp://user:pass@camera/stream`. ESP32 node posts to `/api/sensors`.

## Licences and data

Code: MIT. Datasets are used for evaluation only and are not redistributed:
GMDCSA24 (MIT), UMN crowd (research use), FURG fire (CC0), Home-fire (CC BY-NC 4.0),
DBA-Fire, DeepQuest fire/smoke, PETS-2009 `vtest`. Ultralytics weights are AGPL-3.0;
see `models/README.md`.
