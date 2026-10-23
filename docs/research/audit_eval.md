# Benchmark / evaluation audit — Explainable Edge AI Copilot

Scope: every number in `docs/RESULTS.md` §1–§7, checked against the raw JSONs, the cached
features and independent re-computation. No project files were modified. Audit scripts and raw
outputs are in
`/tmp/claude-0/-home-claude/1a520e14-b6eb-5a1b-8d00-8ea8a0661398/scratchpad/audit_eval/`
(`audit_hash.py`, `audit_metrics.py`, `audit_dup_impact.py`, plus `hash_results.json`,
`metrics_results.json`, `dup_impact.json` and the visual check `dup_pairs.jpg`).

Bottom line: the code that computes the metrics is correct. AUC, AP, F1 and the
attribution checks all reproduce exactly. The problems are in **protocol**:

1. Near-duplicate train/test leakage inflates the fire mAP.
2. The fall, FURG and UMN numbers come from benchmarks that were also used for development and tuning.
3. Several plan criteria are marked as passing when they do not pass, or are missing from the table.
4. No result has a confidence interval, and most sample sizes are very small (n = 10–17).

---

## 1. Leakage

### 1.1 Fire detector: near-duplicates between train and test (major)

Method: I computed an independent 64-bit DCT perceptual hash (pHash) for every image. For each
evaluation image I found the nearest training image by Hamming distance. The evaluation lists are
exactly the ones built by `benchmarks/eval_fire.py` (`eval_sets(400)`, `dba_split`). A distance of 0
means visually the same frame, and `dup_pairs.jpg` confirms this: the mean absolute pixel difference
is 0.0–11 on a 0–255 scale.

| Eval set (n = 400) | NN Hamming = 0 | ≤ 8 | ≤ 10 | Cause |
|---|---|---|---|---|
| Home-fire test vs Home-fire train | **74 (18.5 %)** | **231 (57.8 %)** | 250 (62.5 %) | The upstream dataset's own train/test split contains adjacent video frames |
| DBA-Fire 20 % hold-out vs DBA train | **87 (21.8 %)** | **161 (40.3 %)** | 164 (41 %) | `dba_split()` hashes the **file name**, so consecutive video frames (`vd2000181` / `vd2000178`) land on both sides. 196 of the 197 `vd*` frames in the hold-out have a frame within ±2 in train |
| DeepQuest Neutral (400) vs 600 training negatives | 0 | 0 | 0 | Clean (median distance 20) |
| Home-fire test vs DBA train | 0 | 0 | 5 | Clean |

**Impact on the shipped FP32 model** (same AP code, same detections, split by near-duplicate status):

| Set | Reported mAP@0.5 | Near-duplicate subset | Clean subset (NN > 8) | Clean, NN > 10 |
|---|---|---|---|---|
| Home-fire test | 0.730 | 0.771 (n = 231); 0.814 for exact duplicates | **0.626** (n = 169) | 0.634 (n = 150) |
| DBA hold-out | 0.733 | 0.763 (n = 161) | **0.659** (n = 239) | 0.661 (n = 236); **0.605** after also dropping every video frame (n = 178) |

On clean images the mAP is still above the 0.60 target, but only by 0.03–0.06. The reported
"+0.35 / +0.47 over baseline" is inflated, because the baseline never saw these datasets and so
gets no benefit from the leakage.

### 1.2 Other leakage and selection effects

| Item | Finding | Severity |
|---|---|---|
| Fall, GMDCSA24 | RESULTS §2 says itself that descent memory, low-confidence track continuation, the `tilt_rate` cue, `conf: 0.1` / `track_high_conf` were each "measured on this benchmark". The expert weights and cue design were therefore **developed on the evaluation set**. "Expert weights (no training data used)" is not a held-out result. Scratch scripts (`bias_loso.py`, `fall_percam.py`, `auc_fall.py`, `f_0.1/0.2.json`) show many variants, thresholds and bias grids explored on all 160 clips | Major |
| Fall headline variant | The 0.828 headline is the best of 4 variants, all scored on the same 160 clips (choosing on the test set). Its edge over the untrained expert model is +0.014 F1, 95 % CI [−0.032, +0.062], from 12 discordant videos. That difference is not significant | Major |
| FURG system test | RESULTS §4.2: "Fixes found by this benchmark (all kept)". Ego-motion damping, the flicker weight (0.3 → 0.12), the citation cut-off and the rapid-exit gating were tuned on the 23 FURG videos. The fire-hazard weights also changed against the original config (`smoke_conf` 6 → 8, `fire_conf` 7 → 10, `ref` 0.2 → 0.15) | Major |
| UMN crowd | The "tolerant" labels, the cue set (surge / exodus / divergence replaced speed_ratio / acceleration), one-sided cues and the conformal α were all defined after inspecting UMN. α = 0.01 is the only value that gives "0 FA" under the 1.5 s early tolerance while still missing exactly 1 event (sweep below) | Moderate |
| Demo reel / INT8 | INT8 calibration frames (`data/calib_frames`) come from `demo/demo_campus.mp4`, which contains GMDCSA24, FURG `house1` and UMN test clips. Calibration only sets activation ranges and the INT8 mAP is measured on images, so the effect is small. But the INT8 accuracy claim and the reel's "identical events" are not fully independent | Minor |
| Fire training | Label class order is handled correctly. Source Home-fire/DBA use 0 = fire, 1 = smoke; the training set uses 0 = smoke, 1 = fire. 790 of 800 sampled label files are swapped, and the other 10 are empty negatives. Eval reads the source labels with `("fire","smoke")` and the ONNX with `--onnx-names smoke,fire`, which is consistent. Validation during training = Home-fire val only (1,300 images). `best.pt` = `last.pt` = `models/fire_smoke.pt` (same md5), so no checkpoint was chosen on test | OK |

---

## 2. Metric correctness

| Check | Result |
|---|---|
| `eval_crowd.auc` vs `sklearn.roc_auc_score`, 9 cues (tie fractions up to 99.8 %) plus a synthetic 3-level stress test | Identical (|Δ| = 0) |
| UMN expert frame AUC re-run in memory | 0.8137 (reported 0.814); tolerant 0.9067 (reported 0.907) |
| Fire AP (`ap50`) | Standard VOC all-point AP at IoU 0.5 with greedy matching. The re-run reproduces 0.730 / 0.733 exactly. Capping at max_det = 100 changes nothing (the baseline `.pt` path caps at 100 and the ONNX path does not, but this has no effect) |
| Fall F1 / recall / FAR, all 4 variants | Reproduced exactly: expert 0.814, refit 0.814, bias-tuned 0.828, conformal 0.808 (FAR 8/81 = 0.099) |
| Home-fire `image_fpr@0.25 = 0.667` in `fire_eval.json` | Computed on **3** negative images. Meaningless; it is correctly left out of RESULTS |
| XAI verifier counts | `xai.json`: 349 of **352** detected = **99.1 %**. RESULTS says "99.2 %, 354 cases" |
| Counterfactual count | `xai.json` total = **305**. RESULTS says "100 % of 306" |
| Plan criteria in the headline table | The fire row shows ✅ for mAP only. Plan §5 also requires **image recall ≥ 0.85 at ≤ 10 % FPR on negatives**: measured recall is 0.839 (Home-fire) at **18.5 % FPR** (INT8: 23.8 %). This criterion **fails** and is missing from the table. Saliency criterion "better than random **and centre**": not met for centre (below) |

---

## 3. Claim-by-claim table

| # | Claim (RESULTS) | Verdict | Evidence |
|---|---|---|---|
| 1 | Attribution error 1.8e-15 | ✅ Holds | Exact by construction (additive model) |
| 2 | Template unsupported-claim rate 0 % | ✅ Holds, by construction | The template is generated from the same ledger it is checked against |
| 3 | Verifier catches 99.2 % (354) | ⚠️ Number wrong | 349/352 = 99.1 %, Wilson 95 % CI [97.5, 99.7]. Number drift is the weak class: 70/73 = 95.9 %, CI [88.6, 98.6]. Errors are synthetic perturbations written by the same author as the verifier, so this is a self-consistency check, not evidence about real LLM narrators |
| 4 | False rejects 1.2 % (434) | ✅ Holds | 5/434, CI [0.5, 2.7] % |
| 5 | Counterfactual validity 100 % of 306 / 71 of 73 | ⚠️ Count | Total is 305. Plausibility 231/305 = 75.7 % |
| 6 | UMN 10/11 events, 0 FA, conformal α = 0.01 | ⚠️ Fragile | 10/11, Wilson CI [0.62, 0.98]. 0 FA covers only **199.7 s** of normal video, so the 95 % upper bound is **66 FA/hour**. Without the 1.5 s early tolerance: **2 FA**. α sweep: 0.02 → 11/11, 0 FA (4 without tolerance); 0.005 → 5/11 |
| 7 | UMN frame AUC 0.81 / 0.91 | ⚠️ Post hoc labels | Clip-bootstrap 95 % CI: standard [0.72, 0.96], tolerant [0.85, 0.97]. Per-clip AUC ranges from 0.56 (clip 4) and 0.69 (clip 7) to 1.00. Most of the tolerant gain comes from dropping 206 "empty and still" abnormal frames (0.814 → 0.893). Dropping only the pre-onset second gives 0.826 |
| 8 | Fire mAP 0.730 / 0.733 | ❌ Inflated by leakage | Clean subsets give **0.63 / 0.66** (DBA without video frames: 0.61), §1.1 |
| 9 | Neutral FPR 18.5 % | ✅ Holds; plan criterion fails | Clean set. Recall ≥ 0.85 at ≤ 10 % FPR is not met |
| 10 | FURG 13/17 detected, 1 FA in 4.7 min | ⚠️ Tuned on test, wide CI | Recall CI [0.53, 0.90]. FA rate 12.8/h, Poisson CI [0.3, 71]/h. 1 of 6 negative videos has an FA (CI [3 %, 56 %]) |
| 11 | Fall F1 0.828 LOSO (expert 0.814) | ⚠️ Not held out; variant chosen on test | Bootstrap CI [0.76, 0.89]. The gain over expert is not significant (§1.2). Per-subject F1: 0.76 / 0.94 / 0.86 / 0.69. Bias shift per fold: 1.25 / 0.25 / 1.5 / 0.5 (unstable) |
| 12 | Calibration: empirical FAR 0.099 ≤ α + 1 % | ⚠️ Point estimate only | 8/81 pooled, Clopper–Pearson CI [0.044, 0.185]. Per held-out subject: 0.06 / 0.13 / 0.05 / **0.15**, so two of four subjects break α + 1 %. Video-level ADLs from different subjects are not exchangeable with the calibration set, so "guarantee holds" overstates it |
| 13 | Descent: falls 0.7–0.9 vs lying down 0.2–0.3 body-heights/s | ⚠️ Partly | Per-video peak (with memory): falls median 0.87, IQR [0.67, 1.17]. Lying-down ADLs (n = 14) median 0.59, **IQR [0.29, 1.09]**; all ADLs median 0.56, IQR [0.23, 1.24]. Heavy overlap |
| 14 | Low-confidence track continuation raised F1 0.64 → 0.81 | ? Unverifiable | No stored artefact for 0.64. `f_0.2.json` (min_score 0.2) gives 0.78 |
| 15 | Unconstrained refit learns a negative `ground_time` weight | ✅ Holds | Negative in all 4 folds (−0.37 … −0.78) |
| 16 | GBM on per-clip cue summaries scores lower: LOSO AUC 0.86 vs 0.91 | ⚠️ Gap smaller | Re-run: GBM **0.887**, logistic regression 0.789. Additive model: expert peaks 0.916 (developed on this data), **LOSO-refit peaks 0.895**. Against the comparable LOSO additive model the two are tied |
| 17 | Saliency: Eigen-CAM 0.120 vs random 0.325 / centre 0.122 | ⚠️ Only beats random | n = 10. vs random: Δ = −0.205, CI [−0.34, −0.06], 7/10 wins, Wilcoxon p = 0.037. vs **centre: Δ = −0.002, CI [−0.12, +0.09], 3 wins / 6 losses**. Insertion AUC is **worse** than both controls (0.377 vs 0.389 / 0.408). The plan criterion is not met |
| 18 | Edge FPS 10.7 / 13.0 / 19.3; INT8 −3.3 / −0.8 pt | ✅ / ⚠️ | Not re-timed. The table predates the last fix (this is disclosed). INT8 neutral FPR goes up 18.5 → 23.8 % but is not in the INT8 row |

---

## 4. Confidence intervals (95 %; Wilson for proportions unless noted)

| Quantity | Point | Interval | Method |
|---|---|---|---|
| Fall F1, bias-tuned LOSO | 0.828 | [0.760, 0.888] | Video bootstrap (4,000) |
| Fall F1, expert | 0.814 | [0.741, 0.878] | Video bootstrap |
| Fall F1, conformal | 0.808 | [0.732, 0.875] | Video bootstrap |
| Bias-tuned minus expert F1 | +0.014 | [−0.032, +0.062] | Paired bootstrap |
| Fall recall, bias-tuned | 65/79 = 0.823 | [0.724, 0.891] | Wilson |
| Fall FAR, conformal α = 0.1 | 8/81 = 0.099 | [0.044, 0.185] | Clopper–Pearson |
| Fall video AUC (LOSO-refit peaks / expert peaks) | 0.895 / 0.916 | – | – |
| UMN frame AUC, expert | 0.814 | [0.720, 0.957] | Clip-cluster bootstrap (2,000) |
| UMN frame AUC, tolerant | 0.907 | [0.847, 0.973] | Clip-cluster bootstrap |
| UMN events, conformal | 10/11 | [0.62, 0.98] | Wilson (CP [0.59, 1.00]) |
| UMN FA rate, conformal | 0 in 199.7 s | ≤ 66 /h | Poisson upper bound |
| UMN events, expert threshold | 7/11 | [0.35, 0.85] | Wilson |
| FURG recall | 13/17 | [0.53, 0.90] | Wilson |
| FURG FA rate | 12.8 /h | [0.3, 71] /h | Poisson |
| Verifier detection | 349/352 | [97.5, 99.7] % | Wilson |
| Verifier false rejects | 5/434 | [0.5, 2.7] % | Wilson |
| Eigen-CAM deletion | 0.120 | [0.068, 0.179] | Image bootstrap, n = 10 |
| Centre-control deletion | 0.122 | [0.050, 0.218] | Image bootstrap |
| Fire mAP on clean subsets | 0.626 (HF, n = 169) / 0.659 (DBA, n = 239) | not bootstrapped; n is small | – |

---

## 5. Recommended corrections (exact wording)

**R1. Fire headline row (§1)**, replace with:
> | Fire (images) | mAP@0.5, Home-fire test / DBA-Fire hold-out | 0.730 / 0.733 on the full sets; **0.63 / 0.66 on the subsets with no perceptual near-duplicate in training** (baseline 0.382 / 0.267) | ≥ 0.60 | ⚠️ leakage, see §4.1 |
>
> | Fire (images) | image recall @ ≤ 10 % FPR on negatives | recall 0.84 at **18.5 %** FPR (neutral) | ≥ 0.85 at ≤ 10 % | ❌ |

**R2. §4.1, add after the table:**
> *Near-duplicate check.* A 64-bit perceptual hash shows that 58 % of the Home-fire test images
> and 40 % of our DBA hold-out are within Hamming distance 8 of a training image (18.5 % / 21.8 %
> are pixel-level duplicates). Home-fire's own split contains adjacent video frames, and our DBA
> split hashed file names, so consecutive video frames fell on both sides. On the clean subsets,
> mAP@0.5 is 0.63 (Home-fire, n = 169) and 0.66 (DBA, n = 239; 0.61 without video frames). These
> are the numbers to compare with the target. The baseline never saw either dataset.

Code fix: group `dba_split` by video id (the `vdN` prefix, i.e. `name[:3]`) or by pHash cluster,
de-duplicate the Home-fire test set against train+val (pHash ≤ 8), re-train and re-evaluate.

**R3. Fall row (§1) and §2:**
> | Fall (GMDCSA24) | video F1, leave-one-subject-out | **0.83** (95 % CI 0.76–0.89); untrained expert weights 0.81 — difference not significant | ≥ 0.90 | ❌ |

Replace "Expert weights (no training data used)" with "Expert weights (not fitted, but the cue
design was developed on this dataset)". Add:
> *Caveat.* GMDCSA24 was used both to develop the cues and tracker and to evaluate them, and the
> reported variant is the best of four scored on the same clips. These numbers are optimistic; a
> second fall dataset (e.g. UR-Fall or CAUCAFall) is needed for an unbiased estimate.

**R4. Calibration row and §2 bullet:**
> | Calibration | false-alarm rate vs budget α = 0.1 (fall, held-out subjects) | 0.099 pooled (8/81, 95 % CI 0.04–0.19); per subject 0.06–0.15 | ≤ α + 1 % | ⚠️ |

Replace "Conformal guarantee holds" with "The pooled false-alarm rate matched the budget (0.099 vs
0.10). With 81 normal clips the interval is wide, and two of four held-out subjects exceeded the
budget."

**R5. Crowd §3, event level, append:**
> 0 false alarms were observed in 200 s of normal video (95 % upper bound ≈ 66 per hour). The
> result depends on the 1.5 s early-alert allowance (without it: 2 false alarms) and on α (α = 0.02:
> 11/11 events, 4 false alarms without the allowance). The tolerant labels and α were chosen after
> inspecting UMN.

Frame AUC row: "0.81 (95 % CI 0.72–0.96, clip bootstrap) / 0.91 (0.85–0.97)".

**R6. FURG §4.2, append:**
> These videos were also used to find and fix the failure modes listed below, so 13/17 (95 % CI
> 0.53–0.90) and 1 false alarm in 4.7 min (≈ 13/h, 95 % CI 0.3–71/h) are development-set numbers.

**R7. XAI rows:** "99.1 % (349 of 352)" and "100 % of 305".

**R8. Saliency row and §7:**
> | Saliency | Eigen-CAM deletion AUC vs random / centre controls (n = 10) | 0.120 vs 0.325 / 0.122 — better than random (p = 0.04), **no better than the centre control**; insertion worse than both | better than controls | ❌ |

**R9. §2, gradient-boosting sentence:**
> "…a gradient-boosting classifier on per-clip cue summaries reached a LOSO AUC of 0.89, the same
> as the additive model re-fitted under the same protocol (0.89–0.90), so the black box gave no
> gain."

**R10. §2, descent-memory bullet:** replace "(0.7–0.9) from lying down (0.2–0.3)" with "falls have
a higher peak descent speed (median 0.87 body-heights/s) than lying down (median 0.59), but the
ranges overlap".

**R11.** Remove or substantiate "raised expert F1 from 0.64 to 0.81": no artefact reproduces 0.64.

**R12. Opening paragraph of RESULTS:** replace "Every protocol keeps test data out of
training/tuning" with "Test data is never used to fit weights. However, the fall (GMDCSA24),
crowd (UMN) and fire-video (FURG) benchmarks were also used during development, which makes
those numbers optimistic (see the caveats in each section)."
