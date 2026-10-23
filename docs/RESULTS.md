# Results

All numbers below were produced by the scripts in `benchmarks/` on the build
machine: **2 CPU cores, 7.8 GB RAM, no GPU** (a Raspberry-Pi-5-class budget).
Raw outputs are in `benchmarks/results/*.json`. Every protocol keeps test data
out of training/tuning; where something was tuned, the tuning data is named.

## 1. Headline

| Area | Metric (protocol) | Result | Plan target | Status |
|---|---|---|---|---|
| Explanations | attribution exactness, max abs error (73 real alerts) | **1.8e-15** | < 1e-9 | ✅ |
| Explanations | unsupported-claim rate of template text | **0 %** (≈12 claims / explanation) | 0 % | ✅ |
| Explanations | verifier catches injected narrator errors | **99.2 %** (5 error types, 354 cases) | ≥ 95 % | ✅ |
| Explanations | verifier false rejects on paraphrases | **1.2 %** (434 paraphrases) | ≤ 5 % | ✅ |
| Explanations | counterfactual validity / alerts with a single-cue counterfactual | **100 %** / 71 of 73 | 100 % | ✅ |
| Explanations | explanation + counterfactuals + verification time | **0.54 ms** p50 | ≤ 20 ms | ✅ |
| Crowd (UMN) | events detected, per-camera conformal α = 0.01 | **10 / 11**, **0 false alarms**, median delay 0.8 s | all, ≤ 1 FA/clip | ⚠️ 10/11 |
| Crowd (UMN) | frame AUC, expert weights (standard / label-artifact-tolerant) | 0.81 / **0.91** | ≥ 0.90 | ⚠️ see §3 |
| Fire (images) | mAP@0.5, Home-fire test / DBA-Fire hold-out | **0.730 / 0.733** (baseline 0.382 / 0.267) | ≥ 0.60 | ✅ |
| Fire (video, FURG) | fire videos detected / false alarms on non-fire video | **13 / 17**, **1** in 4.7 min | — | ⚠️ |
| Fall (GMDCSA24) | video F1, leave-one-subject-out | **0.828** (expert, no training: 0.814) | ≥ 0.90 | ❌ see §2 |
| Calibration | empirical false-alarm rate vs budget α = 0.1 (fall, held-out subjects) | **0.099** | ≤ α + 1 % | ✅ |
| Saliency | Eigen-CAM deletion AUC vs random / centre controls | **0.120** vs 0.325 / 0.122 (≈ box control inside the box) | better than controls | ⚠️ §7 |
| Edge | full 3-hazard pipeline, 2 cores | **10.7 FPS** FP32-640, **13.0** INT8, **19.3** FP32-416; cascade +43 % FPS / −33 % CPU | ≥ 8 FPS | ✅ |
| Edge | INT8: size / FPS / fire mAP change | 3.6× smaller / +21 % / −3.3 & −0.8 pt | ≤ 2 pt drop | ⚠️ Home-fire −3.3 |

## 2. Falls — GMDCSA24 (160 clips, 4 subjects, 79 falls)

GMDCSA24 is deliberately hard: falls onto beds, from beds and chairs, people
partly out of frame, and "hard negative" daily activities such as lying down on
a bed, push-ups and sitting on the floor. Pose = YOLO11n-pose ONNX, 15 fps.

| Variant | Acc | Prec | Recall | F1 | Spec | Median delay after annotated onset |
|---|---|---|---|---|---|---|
| Expert weights (no training data used) | 0.831 | 0.894 | 0.747 | 0.814 | 0.914 | 1.18 s |
| LOSO weight re-fit (monotone, L2 to prior) | 0.800 | 0.753 | 0.886 | 0.814 | 0.716 | 0.89 s |
| **LOSO bias tuning (expert weights)** | 0.831 | 0.833 | 0.823 | **0.828** | 0.840 | 1.02 s |
| LOSO conformal threshold, α = 0.1 | 0.825 | 0.881 | 0.747 | 0.808 | 0.901 | 1.07 s |

* **Conformal guarantee holds:** with the alert line set from the training subjects' normal clips at α = 0.1, the held-out false-alarm rate was 0.099.
* **What moved the needle** (each change measured on this benchmark):
  1. *Descent memory*: keep how fast the person went down while they stay down — separates falling (0.7–0.9 body-heights/s) from lying down (0.2–0.3).
  2. *ByteTrack-style association*: fallen people are often detected with low confidence; letting low-confidence detections continue (never start) a track raised expert F1 from 0.64 to 0.81.
  3. *Monotone weight fitting*: re-fitting without sign constraints learned a negative weight for "time motionless on the ground", which would be unsafe in deployment; the constraint keeps every cue's direction physically meaningful.
* **Remaining errors** (`benchmarks/results/fall_gmdcsa24.json`): false alarms are almost all *deliberately lying down on a bed or floor*; misses are mostly *falls onto a bed* (soft, slow) and clips where the head or feet are out of frame so the pose model loses the torso. These are the known hard cases for pose-only fall detection; the triage layer marks low-margin fall alerts as VERIFY rather than ALARM.
* Why we did not reach 0.90: the cue design is now limited by key-point quality (a 2.9 M-parameter pose model on unusual floor postures), not by the fusion — a black-box gradient-boosting classifier on per-clip cue summaries scored *lower* (LOSO AUC 0.86 vs 0.91 for the additive model). Published 96–99 % accuracies are on UR-Fall/Le2i, which are saturated and much easier; cross-dataset results in the literature drop sharply (Le2i F1 100 → 58.8 on in-the-wild video, see landscape report).

## 3. Crowd panic — UMN (11 clips, 3 scenes, 7,739 frames)

The burnt-in "Abnormal Crowd Activity" banner is masked so the model cannot
read the label. Two label artefacts matter: the banner appears up to ~1 s
*after* people start running, and it stays on while the scene is **empty** —
51 % of "abnormal" frames have nobody in view.

| Cue (interpretable) | Frame AUC, standard labels | Frame AUC, tolerant labels* |
|---|---|---|
| surge (peak speed ratio, last 2 s) | 0.857 | 0.934 |
| divergence (scattering outwards) | 0.802 | 0.865 |
| exodus (share of people who left) | 0.738 | 0.825 |
| fused expert model | 0.814 | 0.907 |
| fused, leave-one-scene-out re-fit | 0.750 | 0.839 (scene 1: 0.97, scene 2: 0.77, scene 3: 0.99) |

\*Tolerant = drop the 1 s before the labelled onset and abnormal frames where the scene is empty and still.

**Event level (what an operator sees).** With the expert threshold the system
raised **0 false alarms** but caught 7/11 panics (it misses the calm indoor
scene 2). With **per-camera conformal calibration** (alert line = conformal
quantile, α = 0.01, of smoothed scores on *the other clips of the same camera*)
it caught **10/11** panics with **0 false alarms** and a median delay of 0.8 s
(alerts up to 1.5 s before the banner count as early, given the label lag).
This is the conformal add-on doing exactly its job: the indoor camera's calmer
"normal" earns it a lower alert line.

## 4. Fire and smoke

### 4.1 Detector (images, mAP@0.5, all splits held out)

| Model | Home-fire test (400) | DBA-Fire hold-out (400) | Neutral images FPR @0.25 (400) |
|---|---|---|---|
| Baseline: YOLO11n trained on D-Fire (start point) | 0.382 | 0.267 | 5.8 % |
| **Ours: fine-tuned 4 epochs on CPU, FP32 ONNX** | **0.730** (fire 0.789 / smoke 0.671) | **0.733** (fire 0.589 / smoke 0.877) | 18.5 % |
| Ours, INT8 ONNX | 0.697 | 0.725 | 23.8 % |

The detector alone over-fires on some neutral images (orange light, steam,
haze); that is why it is only one cue of the evidence ledger. At system level
those single-cue detections rarely reach the alert line (next table).

### 4.2 System level — FURG fire videos (full copilot fire branch, 10 fps, first 60 s)

| | Result |
|---|---|
| Fire videos detected | **13 / 17** (median time-to-alert 5.0 s after first annotated flame) |
| False alarms on 6 non-fire videos (4.7 min, incl. hand-held and robot cameras) | **1** |
| Explanations grounded | 100 % of alerts |

Fixes found by this benchmark (all kept): (1) **camera ego-motion** — hand-held
and robot cameras invalidated every "compared with this scene's normal" cue and
caused 4 false alarms; the median optical flow now down-weights those cues'
reliability; (2) flame-colour flicker was over-weighted (red soil triggered it);
(3) cues contributing < 0.1 log-odds are no longer cited; (4) "rapid exit"
needs people in view. Misses are the very small flames of the model-house
videos (`house3/4/6`).

## 5. Explanation quality (73 real alerts from the fall and crowd benchmarks)

| Check | Result |
|---|---|
| Attribution exactness \|bias + Σ contributions − logit\| | max 1.8e-15 |
| Deletion test (set a cue to its reference → logit changes by exactly its contribution) | max error 1.8e-15 |
| Counterfactual validity (re-scored counterfactual flips the decision) | 100 % of 306 |
| Counterfactual plausibility (needed value inside the 5–95 % range seen on normal data) | 76 % |
| Template unsupported-claim rate | 0 % |
| Verifier detection — number drift / false cause / wrong hazard / invented sensor / confidence inflation | 95.9 % / 100 % / 100 % / 100 % / 100 % |
| Verifier false rejects on meaning-preserving paraphrases | 1.2 % |
| Explanation latency | 0.54 ms p50 |

The verifier is what makes an optional on-device LLM narrator safe to add: its
text would be accepted only if it verifies, otherwise the template is shown.
(No LLM could be downloaded on the build machine, so the "narrator" is a
perturbation model of the error types reported for LLM narrators.)

## 6. Edge performance (full pipeline, all three hazards, demo reel, 2 cores)

`benchmarks/bench_latency.py` on the 69 s demo reel (1,032 frames, 960×540, 15 fps;
falls + fire + crowd + sensors all enabled). "CPU-s / video-min" is the CPU
time spent per minute of video — an energy proxy.

| Configuration | Processed FPS | Latency p50 / p95 (ms) | Pose p50 (ms) | CPU-s / video-min | Alerts raised on the reel |
|---|---|---|---|---|---|
| FP32 640, cascade on (default) | 10.7 | 83 / 153 | 59 | 123 | same events in every row* |
| **INT8 640**, cascade on | 13.0 | 69 / 116 | 46 | 90 | identical events, ±0.4 s |
| FP32 416, cascade on | **19.3** | **48 / 76** | 26 | **64** | identical events |
| FP32 640, cascade **off** | 7.5 | 130 / 153 | 58 | 184 | identical events |

\*This table was measured just before the last fix (crowd cues now require
people in view), when the reel still produced a spurious crowd alert during the
fire clip (flame flicker read as "running"). The re-check after the fix
(`benchmarks/results/latency_check.json`) gives 4 alerts: fall 18.8 s (VERIFY,
false), fall 27.3 s (ALARM), smoke/fire 49.5 s (ALARM), crowd 66.4 s (VERIFY).

* The **motion-gated cascade** (heavy models skipped on static frames, detector
  every 2nd frame, keep-alive every 8) raises throughput **+43 %** and cuts CPU
  time **−33 %** with the same alerts.
* **INT8** (static QDQ, head in FP32): models 3.6× smaller (11.6 → 3.6 MB pose,
  10.4 → 3.2 MB fire), **+21 %** FPS, **−27 %** CPU; fire mAP −3.3 / −0.8 points
  (Home-fire / DBA). On this x86 sandbox 416-px FP32 is faster still; on ARM
  boards with int8 dot-product units INT8 is the expected winner.
* Decision time per alert (frame processing + saliency + explanation +
  verification): 90–140 ms at 640 px, 30–78 ms at 416 px; the explanation
  itself takes 0.6 ms (p50).
* Cloud comparison (landscape report §4.5): a cloud round-trip alone is
  ~15–80 ms and a cloud vision-language explanation 0.9–4.6 s per image; here
  the whole decision is on-device and no video leaves the box.

Demo reel timeline (`demo/demo_campus_timeline.json`): walkway (normal) → lying
down on a bed (normal, hard negative) → real fall → small fire starting in a
model house (+ MQ-2 trace) → crowd panic. After the fixes in §4.2 the system
raises: fall at 27.3 s (onset 25.9 s), smoke/fire at 49.5 s (flames from
38.7 s, MQ-2 above normal from 40.7 s), crowd at 66.4 s (UMN panic ~65 s), and
one **false fall alert at 18.8 s** on the lying-down clip, routed as VERIFY
(low margin) — the same failure mode reported for GMDCSA24 in §2.

## 7. Saliency faithfulness (fire/smoke detector, Home-fire test, 10 images)

Deletion (grey-out most salient pixels first; lower = more faithful) and
insertion (reveal from a blurred image; higher = better), 10 steps, score = the
target detection's confidence (D-RISE style).

| Map | Deletion AUC ↓ | Insertion AUC ↑ | Time |
|---|---|---|---|
| random (control) | 0.325 | 0.389 | – |
| centred Gaussian (control) | 0.122 | 0.408 | – |
| **Eigen-CAM, whole frame (on-device, no gradients)** | **0.120** | 0.377 | **4.8 ms** |
| Eigen-CAM re-normalised in the box (shown in console) | 0.038 | 0.509 | 5.5 ms |
| uniform inside the box (control) | 0.030 | 0.519 | – |
| occlusion 6×6 (model-agnostic reference) | 0.029 | 0.464 | 2,739 ms |

**Honest reading.** Eigen-CAM is far better than random (deletion 0.12 vs
0.33) at ~5 ms on the edge CPU, but once the map is restricted to the detection
box it adds essentially nothing over a uniform box (it beat the box control on
only 1 of 10 images), and even the 500× slower occlusion map does no better.
For this detector the *box* is the faithful visual evidence; heat-maps are a
visual aid. This is exactly why the copilot's primary explanation is the
evidence ledger (exact, verifiable) and the heat-map is secondary — a point
worth making, since many "explainable" systems present heat-maps as the
explanation.

## 8. Reproduce

```bash
python benchmarks/extract_fall_features.py --out data/gmdcsa24_lowconf.pkl   # ~35 min on 2 cores
python benchmarks/eval_fall.py
python benchmarks/eval_crowd.py
python scripts/prepare_fire_data.py && python scripts/train_fire.py           # ~17 min / epoch on 2 cores
python benchmarks/eval_fire.py --onnx models/fire_smoke.onnx models/fire_smoke.int8.onnx
python benchmarks/eval_fire_video.py
python benchmarks/eval_xai.py
python benchmarks/eval_saliency.py
python benchmarks/bench_latency.py
```
