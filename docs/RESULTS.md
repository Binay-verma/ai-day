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
| Falls (GMDCSA24) | same, YOLO11s-pose with expert weights (`configs/profiles/fall_accurate.yaml`) / with weights re-fitted per fold | 0.840 (CI 0.77–0.90) / 0.886 (CI 0.83–0.94) | ≥ 0.90 | ❌ not met (the re-fitted weights do not transfer, §8) |
| Calibration | fall false-alarm rate vs budget α = 0.1, held-out subjects | 0.11 (9 / 81, CI 0.06–0.20) with 11n; 0.086 (7 / 81, CI 0.04–0.17) with 11s | ≤ α + 1 % | ⚠️ point estimate only |
| Crowd (UMN) | dispersal events detected, per-camera conformal α = 0.01 | **10 / 11** (CI 0.62–0.98), 0 false alarms in 200 s of normal video (≤ 66 / h upper bound) | all, ≤ 1 FA / clip | ⚠️ development set |
| Crowd (UMN) | frame AUC, standard / label-artefact-tolerant | 0.81 (CI 0.72–0.96) / 0.91 (CI 0.85–0.97) | ≥ 0.90 | ⚠️ tolerant labels post hoc |
| Fire (images, leakage-free) | mAP@0.5, Home-fire test / DBA-Fire scene hold-out | **0.564 / 0.651** (start checkpoint 0.371 / 0.356) | ≥ 0.60 | ⚠️ DBA met, Home-fire not |
| Fire (images, leakage-free) | image recall @ ≤ 10 % FPR on neutral negatives | recall 0.76 (Home-fire) at **15.5 %** FPR (CI 12.3–19.4) | ≥ 0.85 at ≤ 10 % | ❌ |
| Fire (video, FURG) | fire videos detected / false alarms on non-fire video | **15 / 17** (CI 0.66–0.97), median time-to-alert 2.7 s; **0** false alarms in 4.7 min (one-sided 95 % bound ≈ 38 / h) | — | ⚠️ development set |
| Falls, unseen dataset (§10) | UR-Fall, frozen system: falls detected / alerts on daily-activity clips | **27 / 30** with YOLO11s-pose (19 / 30 default); 0 / 5 | — | ✅ with the fall profile |
| External check (§8) | frozen system, unseen sources: fall recall / all-hazard false alarms | **7 / 11** falls (CI 0.35–0.85); **2** false alarms in 7.2 min (CI 1.9–60 / h) | — | ⚠️ honest gap |
| Saliency | Eigen-CAM deletion AUC vs random / centre controls (40 images) | 0.148 vs 0.350 / 0.115: beats random (p < 0.001), **not** the centre control | better than both | ❌ |
| Edge | full 3-hazard pipeline, 2 cores | **12.0 FPS** at 416 px (production config); 7.8 INT8-640; 6.6 FP32-640 | ≥ 8 FPS | ⚠️ met at 416 px only |

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
  estimate is below it. **But the re-fitted weights do not transfer**: fitted on all four
  subjects they put almost all weight on descent speed, and on ordinary walking footage that
  raised 17 false fall alarms in 7 minutes (§8). The shipped profile
  `configs/profiles/fall_accurate.yaml` therefore uses YOLO11s-pose with the **expert**
  weights (F1 0.840 here), which need several cues to agree; the config now enforces that
  (`min_cues: 2`). It costs more CPU (§6).
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
validation images removed). The model was then retrained on CPU with a frozen backbone: 5 epochs,
then 10 more from that checkpoint at half the learning rate (`scripts/train_fire_ft3.sh`). Each
checkpoint was chosen on validation only, and the 15-epoch model was evaluated on test exactly once.
Manifest: `data/fire_splits.json`.

| Model | Home-fire test (1,300) mAP@0.5 | DBA-Fire scene hold-out (665) mAP@0.5 | Neutral FPR @0.25 (400) | ms / image |
|---|---|---|---|---|
| Start checkpoint: YOLO11n trained on D-Fire | 0.371 (fire 0.46 / smoke 0.28) | 0.356 | 5.8 % | 93 |
| Ours, 5 epochs, FP32 ONNX | **0.573** (fire 0.67 / smoke 0.47) | 0.630 (fire 0.56 / smoke 0.70) | 16.0 % | 108 |
| Ours, 5 epochs, INT8 ONNX | 0.563 | 0.615 | 17.8 % | 87 |
| **Ours, 15 epochs, FP32 ONNX (shipped)** | 0.564 (fire 0.68 / smoke 0.45) | **0.651** (fire 0.57 / smoke 0.73) | **15.5 %** (CI 12.3–19.4) | 97 |
| Ours, 15 epochs, INT8 ONNX (shipped) | 0.570 | 0.613 | 18.8 % | 80 |
| *Old v1 model (trained on near-duplicates of these test images)* | *0.725* | *0.663* | *18.5 %* | — |

The v1 row shows how much the leak inflated the earlier headline (0.73): the honest
number for the same recipe is 0.56–0.65. Fine-tuning still adds +0.19 to +0.30 mAP over the
start checkpoint. The detector over-fires on some neutral images (orange light, steam, haze)
and its recall at a 10 % false-positive rate is well below the plan's 0.85, which is why it
is only **one cue** of the evidence ledger: on its own it would not be acceptable as a fire
detector, and the system-level results below show what the fusion makes of it.

*Longer training.* Validation mAP rose from 0.763 (5 epochs) to 0.806 (15 epochs), but on the test
sets the gain was mixed: DBA +0.021, Home-fire −0.009, Home-fire image recall 0.69 → 0.76 at a
slightly lower neutral false-positive rate. The 15-epoch model is shipped because it is better on
three of the four measures; the two are within noise of each other on Home-fire. Most of the
validation gain did not transfer, which is a reminder that validation-set improvements need a
held-out check.

### 4.2 System level: FURG fire videos (full copilot fire branch, 10 fps, first 60 s)

| | 15-epoch detector (shipped) | 5-epoch detector |
|---|---|---|
| Fire videos detected | **15 / 17** (CI 0.66–0.97) | 14 / 17 |
| Median time-to-alert after the first annotated flame | **2.7 s** | 2.8 s |
| False alarms on 6 non-fire videos (4.7 min, incl. hand-held and robot cameras) | **0** (one-sided 95 % bound ≈ 38 / h) | 1 (`icra` robot video) |
| Explanations grounded | 100 % of alerts | 100 % |
| Misses | the very small flames of `house3`, `house4` | + `house6` |

The fused system does much better than the detector alone would suggest: the detector's
false positives on neutral images rarely reach the alert line, because the alarm needs
corroboration (dark-channel smoke density, flicker, persistence, sensors). **Caveat:** these
23 videos were also used to find and fix failure modes (camera ego-motion damping, flicker
weight, citation cut-off, people-in-view gating for rapid exit), so these are
development-set numbers, and 4.7 minutes of non-fire video is far too little to claim a
false-alarm rate.

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

`benchmarks/bench_latency.py` on the 69 s demo reel (1,032 frames, 960×540, 15 fps; falls, fire,
crowd and sensors all enabled), idle machine. "CPU-s / video-min" is CPU time per minute of video
(an energy proxy).

| Configuration | Processed FPS | Latency p50 / p95 (ms) | Pose p50 (ms) | CPU-s / video-min | Alerts on the reel |
|---|---|---|---|---|---|
| FP32 640, cascade on (`campus.yaml` default) | 6.6 | 145 / 271 | 99 | 200 | fall 18.8 s*, fall 27.3 s, fire 47.2 s, crowd 66.4 s |
| INT8 640, cascade on | 7.8 | 123 / 222 | 84 | 150 | same events (±0.3 s) |
| **FP32 416, cascade on (`production.example.yaml`)** | **12.0** | **84 / 140** | 47 | **102** | same events (±0.9 s) |
| YOLO11s-pose 640 (`profiles/fall_accurate.yaml`) | 4.2 | 274 / 389 | 209 | 352 | same events |
| FP32 640, cascade **off** | 4.3 | 228 / 280 | 106 | 310 | same events |

\*The 18.8 s fall alert is the known false alarm on the "lying down on a bed" clip, routed as
VERIFY (§2).

* **The audit fixes cost speed.** The pre-audit build ran at 10.7 FPS at 640 px. The fixes that
  made it safe (the person model now always runs when privacy blurring is on, the fire detector
  runs at least once a second even in a still scene, crowd and flow state is maintained) cost
  about 40 %. At 640 px FP32 the 8 FPS target is no longer met; the 416 px models meet it with
  the same alerts on the reel, so they are the production default. The fire model was trained at
  416 px anyway; the cost of 416 is smaller, more distant people for the pose model.
* The **motion-gated cascade** (heavy models skipped on static frames) gives +53 % FPS and −35 %
  CPU at 640 px with the same alerts.
* **INT8** (static QDQ, detection head kept in FP32, calibrated on training images only): models
  3.3–3.6× smaller, +18 % FPS on this x86 sandbox, fire mAP +0.6 (Home-fire) and −3.8 (DBA) points (§4.1).
  On ARM boards with int8 dot-product units INT8 is the expected winner.
* The **YOLO11s-pose** profile buys the best fall accuracy (§2) for 3× the CPU of the 416 setup;
  it suits a box with one or two cameras, or a mini-PC.
* Cloud comparison (landscape report §4.5): a cloud round-trip alone is ~15–80 ms and a cloud
  vision-language explanation 0.9–4.6 s per image; here the whole decision is on-device and no
  video leaves the box.

## 7. Saliency faithfulness (fire/smoke detector, Home-fire test)

Deletion (grey-out the most salient pixels first; lower = more faithful) and insertion (reveal
from a blurred image; higher = better), 10 steps, score = the target detection's confidence
(D-RISE style), 40 test images, shipped 15-epoch detector.

| Map | Deletion AUC ↓ | Insertion AUC ↑ | Time |
|---|---|---|---|
| random (control) | 0.350 | 0.521 | – |
| centred Gaussian (control) | 0.115 | 0.481 | – |
| **Eigen-CAM, whole frame (on-device, no gradients)** | 0.148 | 0.383 | **6.3 ms** |
| Eigen-CAM re-normalised in the box (shown in console) | 0.060 | 0.585 | 7.8 ms |
| uniform inside the box (control) | 0.040 | 0.600 | – |
| occlusion 6×6 (model-agnostic reference) | 0.036 | 0.571 | 3,520 ms |

**Honest reading.** Eigen-CAM is clearly better than random (deletion 0.148 vs 0.350; better on
29 of 40 images, Wilcoxon p < 0.001) at ~6 ms on the edge CPU. It is **not** better than a plain
centre-bias map (0.148 vs 0.115; better on only 16 of 40, p = 0.16), and inside the detection box
it beats a uniform box on only 4 of 40 images. Even the 500× slower occlusion map is no better
than the box. For this detector the *box* is the faithful visual evidence and heat maps are a
visual aid. That is why the copilot's primary explanation is the evidence ledger (exact,
verifiable), not the heat map; many "explainable" systems present heat maps as the explanation.

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

## 10. Second fall dataset: UR Fall Detection (frozen system, never used in development)

UR Fall Detection (Kwolek & Kepski 2014), camera 0, 30 fall clips and the 5 daily-activity
clips that are mirrored on GitHub (`benchmarks/eval_urfall.py`). Only the RGB half of UR's
depth|RGB frames is used, at 15 fps; the last frame is held 4 s because the clips end about a
second after the impact. Fall onset = first frame labelled "falling" in the official UR CSV.
Nothing was tuned on these clips; each configuration was run once.

| Configuration | Falls detected | ALARM / VERIFY | Median delay after fall onset | Alerts on ADL clips |
|---|---|---|---|---|
| `campus.yaml` (YOLO11n-pose, default) | 19 / 30 (63 %, CI 46–78) | 1 / 18 | 1.9 s | 0 / 5 |
| **`profiles/fall_accurate.yaml` (YOLO11s-pose)** | **27 / 30 (90 %, CI 74–97)** | **19 / 8** | **0.9 s** | 0 / 5 |

This is the strongest evidence in the project for the fall branch, because the data is
independent: on the dataset used for development the larger pose model gained +0.05–0.08 F1;
on unseen data it gains +27 points of recall, detects twice as fast, and is confident enough
for ALARM on two thirds of the falls. The cost is CPU (§6: 4.2 FPS for all three hazards on
2 cores), so the production example runs it only on fall-critical cameras
(`cameras[].pose`, per-camera override). Limits: only 5 negative clips are available on
GitHub, so this says nothing reliable about false alarms; the misses, checked frame by frame: `fall-03` (the person walks right up to the camera,
half out of frame, before falling), `fall-20` (a fall from a chair that ends sitting up
on the floor, not lying) and `fall-21` (the person lands at the bottom edge, mostly out of
frame).

## 11. Privacy filter verification

See [PRIVACY.md](PRIVACY.md#verified-not-assumed-does-the-face-filter-actually-work). An
independent face detector (YuNet) found **0 of 184** faces after the default filter (6
videos, 70 frames). The original Gaussian head blur let it find 26 / 184 (14 %); this
benchmark found that bug and the filter was changed to coarse pixelation.

## 12. Calibration of the stated confidence (falls, YOLO11s-pose, leave-one-subject-out)

`benchmarks/eval_calibration.py`, 160 videos, video-level confidence = peak smoothed
probability, model re-fitted without the held-out subject.

| | ECE (15 bins) | Brier |
|---|---|---|
| as shipped | 0.085 | 0.111 |
| temperature-scaled on training subjects | 0.078 | 0.111 |

The stated confidence is off by about 8 percentage points on average, and temperature
scaling barely helps, so it is not applied; the console shows the confidence together with
the alert line and the ALARM/VERIFY routing rather than as a stand-alone probability.

ROBUSTNESS_SECTION

HARDNEG_SECTION

## 15. What did not work (kept so nobody repeats it)

`benchmarks/results/experiments/README.md`. On GMDCSA24 (LOSO, YOLO11s-pose):

| Idea | Result |
|---|---|
| Monotone GAM shape functions instead of straight-line cue weights (explanations stay exact) | F1 0.863 vs 0.886; best of 8 settings 0.872 |
| One-Euro key-point smoothing before the fall cues | F1 0.859–0.882 vs 0.886 |

Both point the same way: the fall model is limited by what the pose model sees, which §10
confirms on independent data.

## 16. Reproduce

```bash
python benchmarks/extract_fall_features.py --out data/gmdcsa24_lowconf.pkl                 # ~35 min, 2 cores
python benchmarks/extract_fall_features.py --model models/yolo11s-pose.onnx --out data/gmdcsa24_s_lowconf.pkl
python benchmarks/eval_fall.py
python benchmarks/eval_fall.py --cache data/gmdcsa24_s_lowconf.pkl --out benchmarks/results/fall_gmdcsa24_yolo11s.json \
       --profile-out benchmarks/results/experiments/fall_fitted_gmdcsa.yaml --pose-model models/yolo11s-pose.onnx
python benchmarks/eval_crowd.py
python scripts/prepare_fire_data.py && python scripts/train_fire.py --epochs 5 --name ft   # ~20 min / epoch
python benchmarks/eval_fire.py --weights <d-fire start .pt> --onnx models/fire_smoke.onnx models/fire_smoke.int8.onnx
python benchmarks/eval_fire_video.py
python benchmarks/eval_xai.py
python benchmarks/eval_saliency.py
python benchmarks/eval_external.py
python benchmarks/eval_urfall.py && python benchmarks/eval_urfall.py --config configs/profiles/fall_accurate.yaml --out benchmarks/results/fall_urfall_yolo11s.json
python benchmarks/eval_privacy.py
python benchmarks/eval_calibration.py
python benchmarks/eval_robustness.py
python benchmarks/eval_fire_roc.py
python benchmarks/bench_latency.py
```
