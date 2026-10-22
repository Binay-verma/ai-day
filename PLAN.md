# Project plan: Explainable Edge AI Copilot for Smart Emergency Response

Team *local maxima*: Binay Verma, Chetan Jagannath, Bhyresh B S, Dhanush D.

This plan turns the one-page proposal into a system that goes beyond what is
already on the market or in the literature. It is based on the survey in
[`docs/research/landscape.md`](docs/research/landscape.md) (≈95 sources),
[`docs/research/methods.md`](docs/research/methods.md) and
[`docs/research/datasets_models.md`](docs/research/datasets_models.md).

## 1. What already exists (so we do not claim it)

| Already done by others | Examples |
|---|---|
| Plain-language alert summaries | Verkada, Milestone AI plug-in, Ambient.ai, Coram, NVIDIA VSS blueprint (GPU/cloud, black-box text) |
| LLM/VLM "explainable" anomaly detection | Holmes-VAD, VAD-LLaMA, LAVAD, AnomalyRuler (7B+ models, GPU; judged by BLEU/user studies) |
| Grad-CAM / Eigen-CAM heat-maps on YOLO | pytorch-grad-cam, many papers |
| Pose-based fall detection on a Raspberry Pi | many; UR-Fall/Le2i accuracy is saturated (96–99 %) |
| Skeleton-only privacy, motion gating, per-site threshold tuning | Sentinare, Teton, Frigate, Scylla |

**The gap:** nobody we found makes the explanation *the decision itself*
(faithful by construction), checks every sentence of the explanation
against the evidence, gives exact counterfactuals for emergency alerts, or
states a false-alarm guarantee — and all on a CPU-only edge box, across
several hazards at once.

## 2. Our contributions (the add-ons)

| # | Add-on | What it means for an operator | How we measure it |
|---|---|---|---|
| A1 | **Evidence ledger**: additive log-odds fusion of named, unit-bearing cues (torso angle, descent speed, smoke-density rise, crowd speed ratio, MQ-2 ppm…) | Every alert lists *exactly* how much each cue pushed the risk up or down | Attribution exactness error = 0; deletion test |
| A2 | **Grounding verifier** for explanation text | No invented numbers, causes or hazards — even if an LLM paraphrases | Unsupported-claim rate (templates vs perturbed "narrator" text), verifier precision/recall |
| A3 | **Exact counterfactuals + sufficient/necessary reasons** | "It would not have fired if the torso tilt had stayed below 46°" | Validity (100 % flip), sparsity, plausibility |
| A4 | **Conformal false-alarm budget** per hazard/zone | "At most 1 false alarm per N normal windows" stated and tested | Target α vs empirical FAR on held-out normal video |
| A5 | **Cross-hazard corroboration** (smoke ↔ rapid exit) and **verify-vs-alarm deferral** on conflicting evidence | Ambiguous cases go to "please verify", clear ones to "alarm" | FA reduction at matched recall on composited scenarios |
| A6 | **Operator feedback loop**: bounded online intercept update per zone | Dismissing false alarms tunes that camera, never silences it | Simulated feedback study: FAR before/after, cap respected |
| A7 | **Evidence-cited SOP recommendations** from a campus playbook (exits, AED, extinguisher, 112/108/101) | "Bring the AED outside Room B-104 — person motionless 16 s" | Unit tests + every action cites its trigger |
| A8 | **Copilot Q&A** over the evidence ("what if the sensor is wrong?") | Instant, offline, grounded answers | Intent accuracy on a question set |
| A9 | **Edge engineering**: ONNX Runtime, INT8 quantisation, motion-gated cascade, multi-hazard in one process | Runs on a 2-core CPU / Raspberry Pi-class box | FPS, p50/p95 latency, time-to-alert, energy proxy with/without gate, FP32 vs INT8 accuracy |
| A10 | **Privacy by design**: head blurring from pose key-points (no face detector), skeleton-only mode, no raw video leaves the box | DPDP-Act-friendly deployment | Visual check + only evidence ledger is logged |
| A11 | **IoT sensor fusion with health-aware reliability** (ESP32 + MQ-2 + DHT22 + sound) | A dead sensor is ignored and named, not trusted | Unit tests, demo |

## 3. Architecture

```
 camera(s) ──► motion gate ──► YOLO11n-pose (ONNX) ──► tracker ──► fall cues ─────┐
     │                    └──► fire/smoke YOLO (ONNX) + Eigen-CAM feature ──────────┤
     │                         dark-channel smoke density, flame flicker ───────────┤──► additive evidence
     └──► Farneback optical flow ──► crowd speed / coherence / surge ───────────────┤     ledger per hazard
 ESP32 sensors (MQ-2 ppm, temp rise, sound dB) ──► sensor hub (staleness) ──────────┘          │
                                                                                               ▼
     hysteresis ─► conformal threshold ─► ALERT ─► counterfactuals ─► template NLG ─► verifier ─► SOP actions
                                                         │                                         │
                          evidence heat-map (Eigen-CAM / skeleton / flow) ◄────────────────────────┘
                                                         ▼
                 FastAPI + WebSocket operator console (privacy-filtered) ◄─► feedback ─► zone recalibration
```

## 4. Data and models (only GitHub/PyPI reachable from the build box)

* Fall: **GMDCSA24** (160 clips, 4 subjects, falls + hard ADLs like lying on a bed) — cross-subject protocol.
* Crowd: **UMN** unusual crowd activity (11 clips, 3 scenes) — frame AUC + leave-one-scene-out.
* Fire/smoke: GitHub-hosted labelled datasets (see datasets report) — fine-tune YOLO11n, report mAP@0.5 on a held-out split and on negatives.
* Pose: Ultralytics YOLO11n-pose (COCO-17).

## 5. Acceptance criteria (the "benchmark gate")

| Area | Criterion |
|---|---|
| Fall | Video-level F1 ≥ 0.90 cross-subject on GMDCSA24; median detection delay ≤ 1.5 s after annotated fall onset |
| Crowd | UMN frame AUC ≥ 0.90 (expert weights) and ≥ 0.90 leave-one-scene-out; every abnormal clip detected, ≤ 1 false alarm per clip |
| Fire/smoke | mAP@0.5 ≥ 0.60 on held-out split; image-level recall ≥ 0.85 at ≤ 10 % FPR on negatives |
| Explanations | Attribution error < 1e-9; template unsupported-claim rate 0 %; verifier catches ≥ 95 % of injected errors with ≤ 5 % false rejects on paraphrases; counterfactual validity 100 % |
| Saliency | Eigen-CAM deletion AUC better (lower) than random and centre baselines |
| Calibration | Empirical FAR ≤ α + 1 % on held-out normal data |
| Systems | ≥ 8 FPS processed on 2 CPU cores with the cascade; explanation + verification ≤ 20 ms; INT8 accuracy drop ≤ 2 points |
| Engineering | Tests pass, CI config, Docker image, reproducible scripts, README + results report |

## 6. Work loop

Research → plan → build → benchmark → read failures → fix → re-benchmark,
until every row of §5 passes or the remaining gap is explained honestly in
`docs/RESULTS.md`.
