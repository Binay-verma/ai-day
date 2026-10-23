# Results

All numbers below were produced by the scripts in `benchmarks/` on the build
machine: **2 CPU cores, 7.8 GB RAM, no GPU** (a Raspberry-Pi-5-class budget).
Raw outputs are in `benchmarks/results/*.json`. Intervals are 95 % (Wilson for
proportions, bootstrap for F1, Poisson for rates) unless stated.

**How to read these numbers.** Test data is never used to *fit* weights. But the
fall (GMDCSA24), crowd (UMN) and fire-video (FURG) benchmarks were also used while
the cues were being designed, so those numbers are optimistic (each section says
what was tuned on what). Two results are free of that: the leakage-free fire image
benchmark (§4.1) and the frozen-system external check on footage never used in
development (§8). An independent audit of every claim is in
[research/audit_eval.md](research/audit_eval.md); this page applies its corrections.

## 1. Headline

| Area | Metric (protocol) | Result | Plan target | Status |
|---|---|---|---|---|
| Explanations | attribution exactness, max abs error (73 real alerts) | **1.6e-15** | < 1e-9 | ✅ |
| Explanations | unsupported-claim rate of template text | **0 %** (≈ 12 claims / explanation) | 0 % | ✅ by construction |
| Explanations | verifier catches injected narrator errors (8 error types) | **99.8 %** (557 / 558; CI 99.0–99.97) | ≥ 95 % | ✅ synthetic errors |
| Explanations | verifier false rejects on meaning-preserving paraphrases | **0.5 %** (2 / 436; CI 0.1–1.7) | ≤ 5 % | ✅ |
| Explanations | counterfactual validity / alerts with a single-cue counterfactual | **100 %** of 318 / 73 of 73 | 100 % | ✅ |
| Falls (GMDCSA24) | video F1, leave-one-subject-out, shipped YOLO11n-pose | 0.81–0.83 depending on variant (CI ≈ 0.74–0.89) | ≥ 0.90 | ❌ |
| Falls (GMDCSA24) | same, **YOLO11s-pose profile** (`configs/profiles/fall_accurate.yaml`) | **0.886** (CI 0.83–0.94) | ≥ 0.90 | ⚠️ close, not met |
| Calibration | fall false-alarm rate vs budget α = 0.1, held-out subjects | 0.11 (9 / 81, CI 0.06–0.20) with 11n; 0.086 (7 / 81, CI 0.04–0.17) with 11s | ≤ α + 1 % | ⚠️ point estimate only |
| Crowd (UMN) | dispersal events detected, per-camera conformal α = 0.01 | **10 / 11** (CI 0.62–0.98), 0 false alarms in 200 s of normal video (≤ 66 / h upper bound) | all, ≤ 1 FA / clip | ⚠️ development set |
| Crowd (UMN) | frame AUC, standard / label-artefact-tolerant | 0.81 (CI 0.72–0.96) / 0.91 (CI 0.85–0.97) | ≥ 0.90 | ⚠️ tolerant labels post hoc |
| Fire (images, leakage-free) | mAP@0.5, Home-fire test / DBA-Fire scene hold-out | **0.573 / 0.630** (start checkpoint 0.371 / 0.356) | ≥ 0.60 | ⚠️ DBA met, Home-fire not |
| Fire (images, leakage-free) | image recall @ ≤ 10 % FPR on neutral negatives | recall 0.69 (Home-fire) at **16.0 %** FPR (CI 12.7–19.9) | ≥ 0.85 at ≤ 10 % | ❌ |
| Fire (video, FURG) | fire videos detected / false alarms on non-fire video | **14 / 17** (CI 0.59–0.94), median time-to-alert 2.8 s; **1** false alarm in 4.7 min | — | ⚠️ development set |
| External check (§8) | frozen system, unseen sources: fall recall / all-hazard false alarms | **7 / 11** falls (CI 0.35–0.85); **2** false alarms in 7.2 min (CI 1.9–60 / h) | — | ⚠️ honest gap |
| Saliency | Eigen-CAM deletion AUC vs random / centre controls | SAL_ROW | better than both | ❌ |
| Edge | full 3-hazard pipeline, 2 cores | EDGE_ROW | ≥ 8 FPS | ✅ |

## 2. Falls: GMDCSA24 (160 clips, 4 subjects, 79 falls)

GMDCSA24 is deliberately hard: falls onto beds, from beds and chairs, people
partly out of frame, and "hard negative" daily activities such as lying down on
a bed, push-ups and sitting on the floor. 15 fps.

**Caveat.** GMDCSA24 was used both to develop the cues and tracker and to evaluate
them, and several variants are scored on the same clips. "Expert weights" are not
fitted, but the cue design was developed on this dataset. Treat these as optimistic;
§8 gives a frozen-system check on other fall footage.

| Pose model | Variant | Prec | Recall | F1 (95 % CI) | Spec | Median delay |
|---|---|---|---|---|---|---|
| YOLO11n-pose (shipped default) | expert weights | 0.894 | 0.747 | 0.814 (0.74–0.88) | 0.914 | 1.20 s |
| | LOSO monotone re-fit | 0.745 | 0.886 | 0.809 (0.74–0.87) | 0.704 | 0.92 s |
| | LOSO bias tuning | 0.815 | 0.835 | 0.825 (0.76–0.89) | 0.815 | 0.98 s |
| | LOSO conformal line, α = 0.1 | 0.866 | 0.734 | 0.795 (0.72–0.86) | 0.889 | 1.10 s |
| **YOLO11s-pose** | expert weights | 0.887 | 0.797 | 0.840 (0.77–0.90) | 0.901 | 1.14 s |
| | **LOSO monotone re-fit** | **0.886** | **0.886** | **0.886 (0.83–0.94)** | 0.889 | 0.87 s |
| | LOSO bias tuning | 0.831 | 0.873 | 0.852 (0.79–0.91) | 0.827 | 0.94 s |
| | LOSO conformal line, α = 0.1 | 0.901 | 0.810 | 0.853 (0.79–0.91) | 0.914 | 0.92 s |

* **A better pose model helps more than any fusion change.** Swapping the 2.9 M-parameter
  YOLO11n-pose for the 9.9 M YOLO11s-pose (same cues, same fusion) raises every variant.
  With leave-one-subject-out weight fitting it reaches F1 0.886; per held-out subject:
  0.88 / 0.94 / 0.93 / 0.77. The 0.90 target is inside the interval but the point
  estimate is below it. The profile `configs/profiles/fall_accurate.yaml` ships the same
  recipe fitted on all four subjects; it costs more CPU (§6).
* **The re-fit keeps every cue's sign physically meaningful.** Weights are constrained to be
  monotone. Fitted without the constraint, "time motionless on the ground" gets a negative
  weight in all 4 folds (−0.37 … −0.78), which would be unsafe in deployment. With it,
  the fit on 11s features puts most weight on descent speed and body shape; height drop and
  ground time go to 0.
* **Calibration.** With the alert line set from the training subjects' normal clips at
  α = 0.1, the pooled held-out false-alarm rate was 0.11 (11n) and 0.086 (11s). With 81
  normal clips the interval is wide. Per subject the rate ranged 0.05–0.17 (11n) and
  0.04–0.15 (11s), so one or two subjects exceed the budget. Video-level daily activities of
  different people are not exchangeable with the calibration set, so this is evidence, not
  a guarantee.
* **Descent memory.** Falls have a higher peak descent speed than lying down (median 0.87
  vs 0.59 body-heights/s), but the ranges overlap (lying-down IQR 0.29–1.09), which is why
  descent speed is one cue among six.
* **Remaining errors** (`benchmarks/results/fall_gmdcsa24*.json`): false alarms are mostly
  *deliberately lying down on a bed or the floor*. Misses are mostly *falls onto a bed*
  (soft, slow) and clips where head or feet are out of frame. The triage layer marks
  low-margin fall alerts as VERIFY rather than ALARM.
* **Black box comparison.** A gradient-boosting classifier on per-clip cue summaries reached
  a LOSO AUC of 0.89, the same as the additive model re-fitted under the same protocol
  (0.89–0.90), so the black box gave no gain over the explainable model.
* Published 96–99 % accuracies are on UR-Fall/Le2i, which are saturated and easier;
  cross-dataset results drop sharply (Le2i F1 100 → 58.8 on in-the-wild video, landscape report).

## 3. Abnormal crowd movement: UMN (11 clips, 3 scenes, 7,739 frames)

The burnt-in "Abnormal Crowd Activity" banner is masked so the model cannot read the
label. Two label artefacts matter: the banner appears up to ~1 s *after* people start
running, and it stays on while the scene is **empty** (51 % of "abnormal" frames have
nobody in view).

| Cue (interpretable) | Frame AUC, standard labels | Frame AUC, tolerant labels* |
|---|---|---|
| surge (peak speed ratio, last 2 s) | 0.857 | 0.934 |
| divergence (scattering outwards) | 0.802 | 0.865 |
| exodus (share of people who left) | 0.738 | 0.825 |
| fused expert model | 0.814 (CI 0.72–0.96) | 0.907 (CI 0.85–0.97) |
| fused, leave-one-scene-out re-fit | 0.750 | 0.839 (scene 1: 0.97, scene 2: 0.77, scene 3: 0.99) |

\*Tolerant = drop the 1 s before the labelled onset and abnormal frames where the scene is
empty and still. Most of the gain comes from dropping the empty frames (0.814 → 0.893).
Per-clip AUC ranges from 0.56 to 1.00.

**Event level.** With the expert threshold the system raised 0 false alarms but caught only
7/11 events (it misses the calm indoor scene 2). With **per-camera conformal calibration**
(alert line = conformal quantile, α = 0.01, of smoothed scores on the other clips of the
same camera) it caught **10/11** (CI 0.62–0.98) with **0 false alarms**, median delay
0.8 s (range −0.5 to 3.7 s). Caveats: the normal video totals only 200 s (95 % upper bound ≈ 66 false alarms per
hour); the result counts alerts up to 1.5 s before the banner as early (without that
allowance: 2 false alarms); α = 0.02 gives 11/11 with 4 false alarms without the
allowance. The tolerant labels, the cue set and α were chosen after inspecting UMN.

## 4. Fire and smoke

### 4.1 Detector on images, leakage-free protocol (v2)

An audit found the first protocol leaked: 58 % of Home-fire test images were within 8 bits
(perceptual hash) of a training image, and DBA-Fire was split by file name, so frames of
the same video fell on both sides. `scripts/prepare_fire_data.py` (v2) now groups DBA images
into scenes (union-find over pHash distance ≤ 10), splits whole scenes, and removes from
training **every image within 10 bits of any evaluation image** (2,211 training and 723
validation images removed). The model was then retrained on CPU (5 epochs, frozen backbone;
checkpoint chosen on validation only). Manifest: `data/fire_splits.json`.

| Model | Home-fire test (1,300) mAP@0.5 | DBA-Fire scene hold-out (665) mAP@0.5 | Neutral FPR @0.25 (400) | ms / image |
|---|---|---|---|---|
| Start checkpoint: YOLO11n trained on D-Fire | 0.371 (fire 0.46 / smoke 0.28) | 0.356 | 5.8 % | 93 |
| **Ours v2, FP32 ONNX** | **0.573** (fire 0.67 / smoke 0.47) | **0.630** (fire 0.56 / smoke 0.70) | 16.0 % (CI 12.7–19.9) | 108 |
| Ours v2, INT8 ONNX | 0.563 | 0.615 | 17.8 % | 87 |
| *Old v1 model (trained on near-duplicates of these test images)* | *0.725* | *0.663* | *18.5 %* | — |

The v1 row shows how much the leak inflated the earlier headline (0.73): the honest
number for the same recipe is 0.57–0.63. Fine-tuning still adds +0.20 to +0.27 mAP over the
start checkpoint. The detector over-fires on some neutral images (orange light, steam, haze)
and its recall at a 10 % false-positive rate is well below the plan's 0.85, which is why it
is only **one cue** of the evidence ledger: on its own it would not be acceptable as a fire
detector, and the system-level results below show what the fusion makes of it.
FT3_NOTE

### 4.2 System level: FURG fire videos (full copilot fire branch, 10 fps, first 60 s)

| | Result |
|---|---|
| Fire videos detected | **14 / 17** (CI 0.59–0.94); median time-to-alert **2.8 s** after the first annotated flame (v1 model: 13 / 17, 5.0 s) |
| False alarms on 6 non-fire videos (4.7 min, incl. hand-held and robot cameras) | **1** (≈ 13 / h, CI 0.3–71 / h), on the `icra` robot video |
| Explanations grounded | 100 % of alerts |
| Misses | the very small flames of the model-house videos `house3/4/6` |

The fused system does much better than the detector alone would suggest: the detector's
false positives on neutral images rarely reach the alert line, because the alarm needs
corroboration (dark-channel smoke density, flicker, persistence, sensors). **Caveat:** these
23 videos were also used to find and fix failure modes (camera ego-motion damping, flicker
weight, citation cut-off, people-in-view gating for rapid exit), so these are
development-set numbers. With 4.7 minutes of non-fire video the false-alarm interval is very
wide.

## 5. Explanation quality (73 real alerts from the fall and crowd benchmarks)

| Check | Result |
|---|---|
| Attribution exactness \|bias + Σ contributions − logit\| | max 1.6e-15 |
| Deletion test (set a cue to its reference → logit changes by exactly its contribution) | max error 1.8e-15 |
| Counterfactual validity (re-scored counterfactual flips the decision) | 100 % of 318 |
| Counterfactual plausibility (needed value inside the 5–95 % range seen on normal data) | 75 % |
| Template unsupported-claim rate | 0 % (by construction: the template is generated from the ledger it is checked against) |
| Verifier detection: number drift / false cause / wrong hazard / invented sensor / confidence inflation | 98.6 % / 100 % / 100 % / 100 % / 100 % |
| Verifier detection, audit attacks: value swapped between cues / hazard drift / synonym false cause | 100 % / 100 % / 100 % |
| Verifier false rejects on meaning-preserving paraphrases (incl. reordered sentences) | 0.5 % (2 / 436) |
| Explanation + counterfactuals + verification time | ~1 ms p50 |

The verifier is what would make an optional on-device LLM narrator safe: its text would be
shown only if it verifies, otherwise the template is shown. It binds every number to the
cue it describes, checks every hazard word, requires causal claims to name a cue that pushed
toward the alarm, and cuts the playbook strings out exactly so nothing can hide after
"Recommended:". The audit's first pass found it missed 37 % of three new attack types; after
the fix it catches all of them. **Limit:** the errors are synthetic perturbations written by
the same team as the verifier, so this is a consistency check, not evidence about real LLM
narrators (no LLM could be downloaded on the build machine).

## 6. Edge performance (full pipeline, all three hazards, demo reel, 2 cores)

EDGE_SECTION

## 7. Saliency faithfulness (fire/smoke detector, Home-fire test)

SAL_SECTION

## 8. External sanity check: frozen system on footage never used in development

`benchmarks/eval_external.py` runs the shipped `configs/campus.yaml` and models, unchanged,
on clips from sources that were never looked at during development, and reports the first
run. Falls: CAUCAFall sample (5), UR-Fall samples (2), web examples from an open-source fall
repository (4); each fall clip is followed by its last frame held 4 s ("the person stays
down"), because some clips end within a second of the impact. Normal: a UR-Fall daily-
activity clip and 6 Intel IoT DevKit sample videos (walking, aisles, classroom, worker zone),
7.2 minutes in total. No camera was calibrated (commissioning step 4 skipped), which is the
worst case.

| | Result |
|---|---|
| Falls detected | **7 / 11** (CI 0.35–0.85); 1 ALARM, 6 VERIFY |
| Missed | CAUCAFall forward and left falls, one web example, UR-Fall fall-24 (1.9 s clip) |
| False alarms, all hazards | **2** in 7.2 min = 16.6 / h (CI 1.9–60 / h) |
| What they were | a person sitting down fast on UR-Fall ADL (torso rotation 500 °/s, descent 1.6 body-heights/s), routed VERIFY; a group walking towards the camera in a short clip read as abnormal crowd movement ("100 % of the people in view leaving, 5.7× faster than normal, coherence 1.00"), routed ALARM |
| Explanations grounded | 9 / 9 alerts |

**Reading.** This is the most honest number on this page, and it is worse than the
in-domain benchmarks: about two thirds of falls detected, mostly as VERIFY, and a false-alarm
rate that would be unacceptable without calibration. Both false alarms are explained in
terms an operator can check at a glance, which is the point of the evidence ledger. The crowd
false alarm is exactly what per-camera calibration is for: the "normal" speed of a scene is
learned during commissioning, and an uncalibrated camera compares against a generic prior.
Field validation on the real campus cameras is required before any claim about deployed
accuracy.

## 9. Reproduce

```bash
python benchmarks/extract_fall_features.py --out data/gmdcsa24_lowconf.pkl                 # ~35 min, 2 cores
python benchmarks/extract_fall_features.py --model models/yolo11s-pose.onnx --out data/gmdcsa24_s_lowconf.pkl
python benchmarks/eval_fall.py
python benchmarks/eval_fall.py --cache data/gmdcsa24_s_lowconf.pkl --out benchmarks/results/fall_gmdcsa24_yolo11s.json \
       --profile-out configs/profiles/fall_accurate.yaml --pose-model models/yolo11s-pose.onnx
python benchmarks/eval_crowd.py
python scripts/prepare_fire_data.py && python scripts/train_fire.py --epochs 5 --name ft   # ~20 min / epoch
python benchmarks/eval_fire.py --weights <d-fire start .pt> --onnx models/fire_smoke.onnx models/fire_smoke.int8.onnx
python benchmarks/eval_fire_video.py
python benchmarks/eval_xai.py
python benchmarks/eval_saliency.py
python benchmarks/eval_external.py
python benchmarks/bench_latency.py
```
