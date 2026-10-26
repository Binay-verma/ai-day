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
| Explanations | attribution exactness, max abs error (75 real alerts) | **1.3e-15** | < 1e-9 | ✅ |
| Explanations | unsupported-claim rate of template text | **0 %** (≈ 12 claims / explanation) | 0 % | ✅ by construction |
| Explanations | verifier catches injected narrator errors (8 error types) | **100 %** (576 / 576; CI 99.3–100) | ≥ 95 % | ✅ synthetic errors |
| Explanations | verifier false rejects on meaning-preserving paraphrases | **0.5 %** (2 / 443; CI 0.1–1.6) | ≤ 5 % | ✅ |
| Explanations | counterfactual validity / alerts with a single-cue counterfactual | **100 %** of 330 / 75 of 75 | 100 % | ✅ |
| Falls, went down and left the view (§16) | extra falls caught, GMDCSA24 / UR-Fall / external, YOLO11n and YOLO11s | +1 / +1 / 0 and +1 / +3 / 0; **no new false alarm** anywhere | — | ✅ |
| Exit heading (§16) | edge a person leaves by, predicted 2 s ahead (UMN + walking footage, n = 97) | **80 %** (CI 71-87) vs 59 % for the nearest edge | — | ✅ |
| Falls (GMDCSA24) | video F1, shipped YOLO11n-pose, expert weights + fall evidence gate (§2.1) | **0.883** (CI 0.82–0.93), precision 0.970; was 0.814 before the gate | ≥ 0.90 | ❌ close (development set) |
| Falls vs lying down | false fall alarms on GMDCSA24's deliberate lie-down clips (bed/floor), default / accurate profile | **0 / 1** (was 6 / 7 before the gate) | — | ✅ development set |
| Falls (GMDCSA24) | same, YOLO11s-pose with expert weights + gate (`configs/profiles/fall_accurate.yaml`) / with weights re-fitted per fold | 0.843 (CI 0.77–0.90, precision 0.967) / 0.904 (CI 0.85–0.95) | ≥ 0.90 | ❌ not met with the shipped weights; the per-fold re-fit reaches it but its weights do not transfer (§8) |
| Calibration | fall false-alarm rate vs budget α = 0.1, held-out subjects | 0.11 (9 / 81, CI 0.06–0.20) with 11n; 0.086 (7 / 81, CI 0.04–0.17) with 11s | ≤ α + 1 % | ⚠️ point estimate only |
| Crowd (UMN) | dispersal events detected, per-camera conformal α = 0.01 | **10 / 11** (CI 0.62–0.98), 0 false alarms in 200 s of normal video (≤ 66 / h upper bound) | all, ≤ 1 FA / clip | ⚠️ development set |
| Crowd (UMN) | frame AUC, standard / label-artefact-tolerant | 0.81 (CI 0.72–0.96) / 0.91 (CI 0.85–0.97) | ≥ 0.90 | ⚠️ tolerant labels post hoc |
| Fire (images, leakage-free) | mAP@0.5, Home-fire test / DBA-Fire scene hold-out | **0.564 / 0.651** (start checkpoint 0.371 / 0.356) | ≥ 0.60 | ⚠️ DBA met, Home-fire not |
| Fire (images, leakage-free) | image recall @ ≤ 10 % FPR on neutral negatives | recall 0.76 (Home-fire) at **15.5 %** FPR (CI 12.3–19.4) | ≥ 0.85 at ≤ 10 % | ❌ |
| Fire (video, FURG) | fire videos detected / false alarms on non-fire video | **15 / 17** (CI 0.66–0.97), median time-to-alert 2.7 s; **0** false alarms in 4.7 min (one-sided 95 % bound ≈ 38 / h) | — | ⚠️ development set |
| Falls, unseen dataset (§9) | UR-Fall: falls detected / alerts on daily-activity clips | **27 / 30** with the YOLO11s-pose profile, 24 / 30 default (19 and 23 at the first run); 0 / 5 | — | ⚠️ only 5 negative clips; gate and left-view rule added after the first run |
| External check (§8) | unseen sources: fall recall / false alarms in 7.2 min | default 8 / 11 (was 6), fall profile **10 / 11** (was 9); 1 false alarm (crowd, uncalibrated camera), 0 false fall alarms | — | ⚠️ honest gap; gate added after the first run |
| Robustness (§12) | fire mAP kept under 8 corruptions x 3 severities (rPC) / mean person recall | 0.69 / 0.83; collapse in severe noise, flat contrast and darkness, now raised as a low-visibility trouble alert | — | ⚠️ needs IR at night |
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
* **Remaining errors** (`benchmarks/results/fall_gmdcsa24*.json`): before the fall evidence
  gate, false alarms were mostly *deliberately lying down on a bed*; with it (§2.1) they are
  push-ups. Misses are mostly *falls onto a bed* (soft, slow) and clips where head or feet are
  out of frame. The table above is pre-gate; §2.1 has the shipped numbers. The triage layer marks
  low-margin fall alerts as VERIFY rather than ALARM.
* **Black box comparison.** A gradient-boosting classifier on per-clip cue summaries reached
  a LOSO AUC of 0.89, the same as the additive model re-fitted under the same protocol
  (0.89–0.90), so the black box gave no gain over the explainable model.
* Published 96–99 % accuracies are on UR-Fall/Le2i, which are saturated and easier;
  cross-dataset results drop sharply (Le2i F1 100 → 58.8 on in-the-wild video, landscape report).

### 2.1 Falling vs lying down: the fall evidence gate (2026-10-24)

**Problem (webcam field test).** Lying down slowly on a bed raised the fall score past the
line. Root cause: the additive model let *posture* cues alone (torso horizontal, wide body box,
head height dropped, time on the ground) reach the alert line. A person who has lain down
looks exactly like a person who has fallen *once they are down*; the difference is **how they
got there**. Before the gate, 6 of the 7 default-model false alarms on GMDCSA24 (7 of 8 with
the accurate profile) were deliberate lie-downs on a bed.

**Fix.** Two evidence requirements, enforced as explicit, named terms of the additive ledger
(`hazards.fall.gate` in `configs/campus.yaml`, `HazardModel._apply_gate`):

| Requirement | Cues that must together contribute | Minimum | If not met, the explanation says |
|---|---|---|---|
| `fall_motion_gate`: an uncontrolled descent was seen | `descent_speed` + new `hip_drop` (the fastest 1-second drop of the hips during the transition, in standing body-heights) | 0.3 log-odds | "no fast, uncontrolled descent was seen (lying down, not falling)" |
| `fall_posture_gate`: the person ended up lying | `torso_angle` + `aspect_ratio` + `height_drop` | 0.75 log-odds | "the person did not end up lying down (no fall posture)" |

If a requirement fails, the score is capped 0.5 log-odds below the alert line by that rule
term, so attributions still sum exactly (tested) and the dashboard gauge shows *why* it is held
("held below the line: ..."). Exception: someone motionless on the ground for **60 s or more**
passes the motion requirement even if the fall itself was not seen (occluded, out of view,
happened before the camera started) - an unwitnessed long lie matters regardless.

**Effect (expert weights, same clips and protocol as above).**

| Pose model | | Precision | Recall | F1 (95 % CI) | False alarms | of which lie-downs | Falls onto the floor | Falls onto a bed |
|---|---|---|---|---|---|---|---|---|
| YOLO11n-pose (default) | before | 0.894 | 0.747 | 0.814 (0.74–0.88) | 7 | 6 | 50 / 65 | 9 / 14 |
| | **with gate** | **0.970** | **0.810** | **0.883 (0.82–0.93)** | **2** | **0** | **57 / 65** | 7 / 14 |
| YOLO11s-pose (profile) | before | 0.887 | 0.797 | 0.840 (0.77–0.90) | 8 | 7 | 55 / 65 | 8 / 14 |
| | **with gate** | **0.967** | 0.747 | 0.843 (0.77–0.90) | **2** | 1 | 53 / 65 | 6 / 14 |

Recall *rose* for the default model: the hip-drop cue adds real evidence to genuine falls.
The remaining false alarms are push-ups (a fast, real descent to a horizontal posture) and one
lie-down onto a bed with the larger model.

**Second webcam test (same day).** The slow lie-down raised nothing and the drop onto a mat
raised an ALARM. Two short-lived fall ALARMs 7 s after start-up cited impossible values (hips
dropping 1.39 body-heights at 5 body-heights/s): the tracks had never been seen upright, so the
current, horizontal box served as the body-height reference. The hip-drop cue had no reliability
discount for that case (descent speed did); it now has the same one. Cost on GMDCSA24: one fall
each (table above includes it), one push-up false alarm fewer with YOLO11n-pose. Every time a gate
holds a would-be alert below the line, a `hazard_held` entry with the reason and the score
without the gate is written to the audit trail (and the session export), so a quiet lie-down
leaves evidence instead of nothing.

**Third webcam test: lying down on the floor from standing.** Lying down on the floor in 2-3 s
still alarmed. The hip-drop cue measured how *far* the hips went down over 3 s, and a
controlled lie-down from standing drops them as far as a fall. It now measures the fastest drop
within any 1 s (a speed), looked up over the 2 s before the low posture is reached and kept as
long as descent speed is (3 s). A first version (last 1 s only, minimum 0.5) cost falls on every
benchmark (UR-Fall 23 → 21, external 8 → 5 with the default model), so it was not shipped; the
look-back, the 3 s memory and the minimum of 0.3 were then chosen on GMDCSA24 and on simulated
lie-downs (controlled 2-6 s lie-downs on the floor or onto a bed must be held, a 0.5-1 s fall
must alarm; regression test `test_lying_down_on_the_floor_from_standing_is_not_a_fall_but_a_fast_drop_is`).
The final version loses nothing anywhere: GMDCSA24 63 → 64 falls (YOLO11n) and 58 → 59
(YOLO11s) with the same 2 false alarms each; UR-Fall 23 / 24 unchanged; external 8 / 11 unchanged
and 9 → 10 / 11 with the profile. The UR-Fall and external clips were looked at while finding
why the first version lost falls (fall-06, fall-23), so this change too is disclosed as not a
first run.

The inherent limit is **falling onto a bed from
sitting on it**: physically it is the same movement as lying back onto the bed, and about half
of those 14 clips are now not alerted. Thresholds were chosen on GMDCSA24 only (the posture
minimum: 0.75 lost nothing there, 1.0 and 1.5 each lost a fall).

**Disclosure.** The gate was added after the first runs of §8 and §9, and the posture
requirement was prompted by one new false alarm it caused on the external set (a shopper in a
store aisle, "torso rotating 500 deg/s, a rapid descent" while upright). Its threshold was then
set on GMDCSA24 alone. §8 and §9 are therefore no longer first runs for this change; their
before/after numbers are both reported.

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

## 5. Explanation quality (75 real alerts from the fall and crowd benchmarks)

| Check | Result |
|---|---|
| Attribution exactness \|bias + Σ contributions − logit\| | max 1.3e-15 |
| Deletion test (set a cue to its reference → logit changes by exactly its contribution, plus the named gate term when the deletion makes a fall evidence gate fire: 14 of 511 deletions) | max error 1.8e-15 |
| Counterfactual validity (re-scored counterfactual flips the decision) | 100 % of 330 |
| Counterfactual plausibility (needed value inside the 5–95 % range seen on normal data) | 85 % |
| Template unsupported-claim rate | 0 % (by construction: the template is generated from the ledger it is checked against) |
| Verifier detection: number drift / false cause / wrong hazard / invented sensor / confidence inflation | 100 % / 100 % / 100 % / 100 % / 100 % |
| Verifier detection, audit attacks: value swapped between cues / hazard drift / synonym false cause | 100 % / 100 % / 100 % |
| Verifier false rejects on meaning-preserving paraphrases (incl. reordered sentences) | 0.5 % (2 / 443) |
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
worst case. (These sources were used only for this check; after it, one configuration was
rejected - see below - so for the fall profile they are no longer completely unseen.)

**Correction (2026-10-24).** UR-Fall videos store depth and RGB side by side; the first run fed
the whole double frame to the models. With only the RGB half (as in §9) the numbers below are
the corrected first runs of each configuration.

| | Default (`campus.yaml`, YOLO11n-pose) | Fall profile (YOLO11s-pose, expert weights) | *Rejected: YOLO11s-pose with GMDCSA-fitted weights* |
|---|---|---|---|
| Falls detected | 6 / 11 (CI 0.28–0.79); **8 / 11** with the fall evidence gate | 9 / 11 (CI 0.52–0.95); **10 / 11** with the gate | *9 / 11* |
| False fall alarms on 7.2 min of normal video | 0 | **0** | ***17*** |
| Other false alarms | 1 crowd | 1 crowd | 1 crowd |
| Explanations grounded | 7 / 7 (9 / 9 with the gate) | 10 / 10 (12 / 12 with the gate) | 28 / 28 |

**What the false alarms were.** The crowd alert in every column is the same: a group walking
towards the camera in a short clip, read as abnormal crowd movement ("100 % of the people in
view leaving, 5.7x faster than normal, coherence 1.00"), routed ALARM. That is exactly what
per-camera calibration is for: the "normal" speed of a scene is learned during commissioning,
and an uncalibrated camera compares against a generic prior. The 17 fall alarms of the fitted
weights all cite "a rapid descent" alone - people walking into or out of frame. Weights fitted
on one dataset had learned a single-cue shortcut that the expert weights (which need several
cues to agree) do not have, so the shipped profile keeps the expert weights and the config now
rejects any fall model in which one cue can reach the alert line (`min_cues: 2`).

**After the fall evidence gate (§2.1).** The first version of the gate (motion requirement
only) raised 8 / 11 falls but added one false fall alarm with the default model: a shopper in a
store aisle read as "the torso rotating 500 deg/s, a rapid descent of 2.1 body-heights/s" while
still upright (VERIFY, 65 %). That prompted the posture requirement (the person must end up
lying), whose threshold was then set on GMDCSA24 alone. Final (after the third webcam fix, §2.1): default 8 / 11, profile 10 / 11,
no false fall alarm with either. Because this change was made after seeing these clips, the gate
rows are not first runs.

**Reading.** These are the most honest numbers on this page and they are worse than the
in-domain benchmarks: the default system finds about half of these falls, the fall profile
about four in five, and no camera was calibrated. Every alert is explained in terms an operator
can check at a glance, which is how the bad profile was caught. Field validation on the real
campus cameras is required before any claim about deployed accuracy.

## 9. Second fall dataset: UR Fall Detection (frozen system, never used in development)

UR Fall Detection (Kwolek & Kepski 2014), camera 0, 30 fall clips and the 5 daily-activity
clips that are mirrored on GitHub (`benchmarks/eval_urfall.py`). Only the RGB half of UR's
depth|RGB frames is used, at 15 fps; the last frame is held 4 s because the clips end about a
second after the impact. Fall onset = first frame labelled "falling" in the official UR CSV.
Nothing was tuned on these clips; each configuration was run once.

| Configuration | Falls detected | ALARM / VERIFY | Median delay after fall onset | Alerts on ADL clips |
|---|---|---|---|---|
| `campus.yaml` (YOLO11n-pose, default), first run | 19 / 30 (63 %, CI 46–78) | 1 / 18 | 1.9 s | 0 / 5 |
| `profiles/fall_accurate.yaml` (YOLO11s-pose, expert weights), first run | 23 / 30 (77 %, CI 59–88) | 3 / 20 | 2.2 s | 0 / 5 |
| `campus.yaml` + fall evidence gate (§2.1, shipped) | **23 / 30** (77 %, CI 59–88) | **11 / 12** | 1.5 s | 0 / 5 |
| `profiles/fall_accurate.yaml` + fall evidence gate | 24 / 30 (80 %, CI 63–90) | 14 / 10 | 1.2 s | 0 / 5 |
| `campus.yaml` + gate + went-down-and-left-the-view (§16, shipped) | **24 / 30** (80 %, CI 63–90) | 11 / 13 | 1.5 s | 0 / 5 |
| **`profiles/fall_accurate.yaml` + gate + left-view (shipped)** | **27 / 30 (90 %, CI 74–97)** | **14 / 13** | **1.1 s** | 0 / 5 |
| *rejected: YOLO11s-pose, GMDCSA-fitted weights* | *27 / 30* | *19 / 8* | *0.9 s* | *0 / 5 - but 17 false fall alarms on walking footage (§8)* |

The evidence gate *helped* here, although it was built to remove false alarms: the hip-drop cue
adds evidence to real falls, so more of them clear the line, sooner and with ALARM rather than
VERIFY (default: 1 → 11 ALARMs). The gate was added after the first runs, so its rows are not
first runs (§2.1 disclosure). Before the gate, the larger pose model added 4 falls (+13 points of recall) on independent data. The fitted
weights look even better here (27 / 30, faster, more ALARMs) - and this is the trap: UR-Fall
has only 5 negative clips, so it cannot show false alarms; the external walking footage (§8)
showed 17. A benchmark without enough negatives rewards a trigger-happy model. The profile costs
CPU (§6: 4.2 FPS for all three hazards on 2 cores), so the production example runs it only on
fall-critical cameras (`cameras[].pose`, per-camera override). Limits: only 5 negative clips are available on
GitHub, so this says nothing reliable about false alarms. Before the gate, the 7 misses of the profile,
checked frame by frame, share one pattern: the person ends up close to the camera at the
bottom edge of the image, partly cut off, curled up or lying along the viewing direction
(foreshortened, so the body still looks upright in 2-D: `fall-01`, `-05`, `-19`, `-21`, `-27`),
plus one fall from a chair that ends sitting up (`fall-20`) and one that ends with the legs
raised (`fall-04`). Camera placement that sees the whole floor (OPERATIONS.md, commissioning
step 2) matters as much as the model. With the gate the profile misses 6: `fall-19`, `-20`, `-21`, `-27`
remain, `fall-01`, `-04`, `-05` are now found, and `fall-06` and `-11` are newly missed (their
descent was not fast enough to pass the motion requirement before the person was down). With the went-down-and-left-the-view rule (§16) the profile misses only 3 (`fall-06`, `-11`,
`-20`): `fall-19`, `-21` and `-27` are now raised as VERIFY "possible fall out of view" alerts.

## 10. Privacy filter verification

See [PRIVACY.md](PRIVACY.md#verified-not-assumed-does-the-face-filter-actually-work). An
independent face detector (YuNet) found **0 of 184** faces after the default filter (6
videos, 70 frames). The original Gaussian head blur let it find 26 / 184 (14 %); this
benchmark found that bug and the filter was changed to coarse pixelation.

## 11. Calibration of the stated confidence (falls, YOLO11s-pose, leave-one-subject-out)

`benchmarks/eval_calibration.py`, 160 videos, video-level confidence = peak smoothed
probability, model re-fitted without the held-out subject.

| | ECE (15 bins) | Brier |
|---|---|---|
| as shipped (with the fall evidence gate) | 0.095 | 0.110 |
| temperature-scaled on training subjects | 0.090 | 0.110 |
| *before the gate, as shipped* | *0.085* | *0.111* |

The stated confidence is off by about 10 percentage points on average. The evidence gate
worsened ECE slightly (0.085 → 0.095; Brier the same): a held clip's score is capped just below
the line (~48 %), whether or not it was a fall, which puts mass in one mid bin. Temperature
scaling barely helps, so it is not applied; the console shows the confidence together with
the alert line and the ALARM/VERIFY routing rather than as a stand-alone probability.

## 12. Corruption robustness

`benchmarks/eval_robustness.py`: the ImageNet-C protocol (Hendrycks & Dietterich 2019; mPC /
rPC from Michaelis et al. 2019) on a CCTV-relevant subset plus low light, severities 1 / 3 / 5.
Fire: mAP@0.5 on 300 leakage-free Home-fire test images and false-positive rate on 200 neutral
images. People: recall of the clean-frame person detections (pseudo ground truth) on 76 frames
from GMDCSA24 and the Intel sample videos.

| Corruption | Fire mAP s1 / s3 / s5 (clean 0.578) | Person recall s1 / s3 / s5 |
|---|---|---|
| gaussian noise | 0.48 / 0.27 / 0.15 | 0.97 / 0.86 / **0.22** |
| motion blur | 0.55 / 0.49 / 0.31 | 0.99 / 0.96 / 0.79 |
| defocus blur | 0.55 / 0.44 / 0.28 | 1.00 / 0.97 / 0.90 |
| fog | 0.47 / 0.38 / 0.26 | 0.99 / 0.97 / 0.94 |
| contrast | 0.52 / 0.32 / **0.00** | 0.97 / 0.94 / **0.18** |
| JPEG compression | 0.57 / 0.54 / 0.43 | 0.98 / 0.90 / 0.58 |
| pixelate | 0.58 / 0.57 / 0.56 | 0.98 / 0.97 / 0.78 |
| low light (night without IR) | 0.57 / 0.33 / **0.02** | 1.00 / 0.91 / **0.10** |
| **mean over all (mPC)** | **0.40, rPC 0.69** | **0.83** |

Person detection (and so fall detection) holds up well to blur, fog, compression and moderate
darkness, but collapses in three conditions: severe sensor noise, a washed-out flat image, and
near-darkness. Fire detection degrades earlier, notably with noise and fog, which is another
reason the fire alarm needs corroboration from flicker, smoke density and sensors.

**What the system does about it.** A model that cannot see must not stay silent. The camera
health check now raises a **low-visibility** trouble alert ("image too dark, flat or noisy for
reliable analysis") when the image's grey-level spread falls below 5 or its estimated noise
passes 32 for 10 s; thresholds were set from this benchmark (clean CCTV: spread 40-50, noise
at most 8.6). On the benchmark frames it flags 0 / 76 clean, moderate-noise, moderate-dark,
moderate-contrast and fog frames, where recall stays at 0.86 or more, and 76 / 76 at severe
noise and severe contrast; severe low light is caught as **blackout** on 74 / 76 frames. Night
operation therefore needs IR illumination or a low-light camera; the trouble alert is what
tells the operator when that is missing.

## 13. Fire detector operating point and hard-negative mining (not adopted)

`benchmarks/eval_fire_roc.py`: image score = highest fire/smoke confidence; positives =
leakage-free Home-fire test images (1,285), negatives = DeepQuest neutral (400).

| Model | ROC AUC | Recall at 10 % FPR | FPR at 85 % recall | Recall / FPR at conf 0.25 |
|---|---|---|---|---|
| **shipped (15 epochs)** | 0.862 | **0.672** | 0.285 | 0.756 / 0.155 |
| + hard-negative fine-tune (4 epochs) | 0.861 | 0.662 | 0.350 | 0.693 / 0.118 |

The plan's criterion (recall ≥ 0.85 at ≤ 10 % FPR) is far from met: at a 10 % false-positive
budget the detector finds two thirds of the fire images. To attack that, 188 confounder images
that the detector fired on (warm room lighting, red decor, aurora, sunsets; mined from the cair
"Normal Images" set, 1 near-duplicate of an evaluation image removed, and **3 images showing a
lit fireplace or smoke removed after looking at every one** - a lit fireplace is real fire) plus
102 clean ones were added and the model was fine-tuned for 4 epochs, checkpoint rule fixed in
advance (last epoch). It lowered false positives at the default threshold (15.5 % → 11.8 %)
but lost the same amount of recall: the ROC curve did not move, so by the rule set before the
test it was **not adopted** (`scripts/mine_hard_negatives.py`, `scripts/train_fire_hn.sh`).
Shifting the threshold would give the same trade-off for free. Reading: the detector's
confusions are not a lack of negative examples; a better fire *image* detector needs more
positive indoor fire data or a larger model - and the system-level answer stays the evidence
ledger, where a single detection of warm light does not reach the alert line without
corroboration (FURG: 0 false alarms, §4.2).

## 14. What did not work (kept so nobody repeats it)

`benchmarks/results/experiments/README.md`. On GMDCSA24 (LOSO, YOLO11s-pose):

| Idea | Result |
|---|---|
| Monotone GAM shape functions instead of straight-line cue weights (explanations stay exact) | F1 0.863 vs 0.886; best of 8 settings 0.872 |
| One-Euro key-point smoothing before the fall cues | F1 0.859–0.882 vs 0.886 |

Both point the same way: the fall model is limited by what the pose model sees, which §9
confirms on independent data.

## 15. Reproduce

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
python benchmarks/extract_walk_dets.py && python benchmarks/eval_exits.py      # §16 heading prediction
python benchmarks/eval_left_view.py && python benchmarks/eval_left_view.py --cache data/gmdcsa24_s_lowconf.pkl --out benchmarks/results/left_view_gmdcsa24_yolo11s.json
python benchmarks/bench_latency.py
python benchmarks/eval_frame_rate.py                                                     # §17 frame rate
node tests/e2e/browser_camera.e2e.js http://127.0.0.1:7861 <pw> fakecam.y4m 72 out/     # §17 real browser
```

## 16. Site awareness: exits, headings, "went down and left the view"

Four cross-camera features were built for emergencies, without identity or person re-identification;
each was kept only if its numbers held up. Raw outputs: `benchmarks/results/exits.json`,
`benchmarks/results/left_view_*.json`.

**Shared building block: where people leave the view, and where they are heading**
(`perception/exits.py`, `benchmarks/eval_exits.py`). Ground truth is the edge through which each
track actually left; the prediction is made 0.5-3 s earlier. Three predictors were compared on UMN;
the blended one (direction of motion weighted by speed, minus distance to the edge) was chosen on UMN
and then checked on walking footage (Intel / OpenCV sample videos, detections cached by
`extract_walk_dets.py`).

| Ahead of the exit | Heading predictor, UMN (dispersals) | Walking footage | Pooled (n) | "Nearest edge" baseline, pooled |
|---|---|---|---|---|
| 0.5 s | 84 % | 100 % | **88 %** (113) | 88 % |
| 1 s | 81 % | 89 % | **83 %** (104), CI 74-89 | 76 % |
| 2 s | 77 % | 88 % | **80 %** (97), CI 71-87 | 59 %, CI 49-68 |
| 3 s | 78 % | 75 % | **77 %** (87), CI 67-85 | 59 % |

### Kept: 3. Went down and left the view
A person who was going down leaves the camera's view at an edge before the fall can be confirmed (it
rolls or crawls out of the picture, or ends up cut off at the bottom). Rule, chosen on GMDCSA24 only:
left through an edge, low posture when last seen, an uncontrolled descent seen in the last 1.5 s (the
same evidence the fall motion gate needs), fall score >= 10 % before the gates, and no fall alert yet
for that person. Always VERIFY, names the place beyond that edge (`cameras[].edges`), and the ledger is
unchanged (a zero-weight `left_view` rule term).

| Data (pipeline replay) | Without | With | False alarms |
|---|---|---|---|
| GMDCSA24, YOLO11n, clips as recorded | 64 / 79 (F1 0.883) | **65 / 79 (F1 0.890)** | 2 -> 2 |
| GMDCSA24, YOLO11s, clips as recorded | 59 / 79 (F1 0.843) | **60 / 79 (F1 0.851)** | 2 -> 2 |
| GMDCSA24, person stays visible 4 s, YOLO11n / YOLO11s | 67 / 60 | **68 / 62** (F1 0.913 / 0.867) | 2 -> 2 |
| UR-Fall (held out for this rule), YOLO11n / YOLO11s | 23 / 24 of 30 | **24 / 27 of 30** | 0 / 5 ADL clips |
| External clips (held out), YOLO11n / YOLO11s | 8 / 10 of 11 | 8 / 10 of 11 (unchanged) | 0 false fall alarms in 7.2 min |

The UR-Fall falls it recovers (fall-13; fall-19, -21, -27) are the documented weak case of §9: the
person ends up at the bottom edge, partly out of view. A first version that removed every detection
at the end of each clip looked much better (F1 0.914) because it made people lying near the edge
"leave"; that protocol was unrealistic and discarded. Cost: 0.4 ms per frame with 20 people in view.

### Kept: 1. Safe-exit guidance and 2. Crowd heading into danger
`site.py`. When smoke / fire reaches the alert line on a camera that watches an exit
(`site.exits.<name>.cameras`), the exit is marked unsafe (for `hazard_ttl_s` after the hazard clears);
recommendations route people to the exits still safe and add "Do NOT send people to <exit>: smoke /
fire seen by camera <id> (<score>, alert line <line>)"; if no listed exit is safe, they say so. An
"Exit unsafe" advisory names the new routes per zone. When at least 3 people are heading (predictor
above) for an unsafe exit or a zone with active fire / smoke, a CRITICAL "Crowd heading into danger"
alert names the place, the evidence and the safe alternative. This is decision logic on top of measured
inputs, so it is checked by scenario tests rather than a dataset (`tests/test_site.py`, 17 scenarios:
blocked exit, time-to-live, no safe exit, fall alerts not mentioning exits, rate limits, fewer than 3
people, a crowd too far away, a zone instead of an exit, the API, review and search of the advisories,
and that every routed recommendation still passes the explanation verifier). A labelled demo scenario
(`--demo`, **Site scenario** button) drives the real logic on one laptop. Limit: no multi-camera site
footage was available, so the end-to-end chain (camera -> exit -> crowd) has not been measured on real
video.

### Discarded: 4. Exit flow, capacity and time-to-clear
Counting people leaving through each exit (to estimate flow per minute, capacity and time to clear a
zone) could not be validated. On UMN the detector finds only 2-6 of the 12-15 people present at
dispersal onset (320x240, motion blur), so 17 exits were counted against 54 people counted by hand in
the outdoor clips (mean error 7.4 people per clip), and the plaza clips end ~1.3 s after onset. The
counting code is kept for the benchmark only; no flow, capacity or time-to-clear number is shown to
operators.

## 17. "Use my camera" (browser camera) check

On the hosted demo a judge can analyse their own laptop camera (`/ws/camera`, `BrowserCamera` in
`cameras.py`). The browser sends JPEG frames with their capture time, one frame in flight, so the rate
follows the server. Each frame goes through the same pipeline as a site camera. The annotated,
privacy-filtered frame goes back to that browser only, and alerts go to the console as usual. Raw outputs:
`benchmarks/results/browser_camera_e2e.json`, `benchmarks/results/frame_rate_gmdcsa24.json`.

**End to end in a real browser.** Headless Chromium used a video file as its webcam: a GMDCSA24 clip of
someone lying down on a bed, then one of a walk and a fall, each held still, looping every 33.8 s. It ran
against the cloud-demo configuration on 2 vCPU, with the demo reel running
(`tests/e2e/browser_camera.e2e.js`, `benchmarks/bench_browser_camera.py`).

| Run | Frame rate | Falls alerted | Lie-downs alerted |
|---|---|---|---|
| Chromium; browser, its fake camera and server share 2 vCPU | 9.4 FPS | 2 of 2 (ALARM, VERIFY) | 0 of 2 |
| Thin client, server-limited (one camera, demo reel slowed to 2 FPS) | **11.6 FPS** (round trip 78 ms median) | 2 of 2 (ALARM, VERIFY) | 0 of 2 |

Each fall alert came about 2 s after the fall began. The first run found a real bug, now fixed. When
people were not moving towards an edge, the heading summary carried an ETA of `Infinity`, which browsers
reject as JSON, so those frames were dropped by the console. This affected the main live feed too.

**Frame rate matters for falls.** In many GMDCSA24 falls the detector catches the fast part in only one
or two frames, so dropping frames removes the evidence the motion gate needs. Production pipeline replay,
rule on, 79 falls and 81 ADL clips (`benchmarks/eval_frame_rate.py`):

| Frames kept | ~FPS | Falls alerted, clip / hold | False alarms, clip / hold |
|---|---|---|---|
| all | 15 | 65 / **68** (86 %) | 2 / 2 |
| 3 of 4 | 11 | 60 / 64 (81 %) | 1 / 1 |
| 1 of 2 | 7.5 | 53 / 60 (76 %) | 2 / 2 |
| 1 of 3 | 5 | 50 / 59 (75 %) | 0 / 2 |

So the browser camera is built for frame rate:
* It runs fall and fire/smoke only. Crowd is abnormal movement of many people, which a laptop camera seeing
  1-3 people cannot show, and its optical flow costs 16 of 73 ms per frame.
* The fire detector runs on every 3rd frame.
* The browser sends up to 15 FPS.
* On the hosted demo, one session runs at a time, and recorded demo files slow to 2 FPS while it is live.
  Live (RTSP) cameras are never slowed.

On 2 vCPU that gives 11.6 FPS. A 4-core Codespace is recommended for judging. The console shows the frame
rate and says when it is below 13 FPS that fast falls can be missed.

**Tried, not adopted: counting frame-based evidence in seconds.** Three things are counted in frames:
standing observations before motion evidence is fully trusted, the per-frame smoothing weight, and frames
below the line before an alarm clears. Counting them in seconds below 14 FPS helped only at ~5 FPS
(+6 falls, clip). At ~11 FPS it caught the same falls with one more false alarm. On the production
benchmarks, where 29 of the 160 cached clips are stored at 7.5 FPS, the configured model was unchanged,
but two research variants moved:

* LOSO-fit, YOLO11s: 71 falls / 7 false alarms became 70 / 8.
* Calibration error (ECE): 0.095 became 0.105.

It was removed, and published results are byte-identical to before.
