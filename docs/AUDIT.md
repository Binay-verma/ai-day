# Audit summary and production-readiness verdict

This page answers one question: *is this project at industry standard, and what
does "production-ready" honestly mean for it?* It summarises four independent
reviews run on the code base (each by a separate reviewer that had not written
the code), what they found, what was fixed, and what can only be closed by the
institution that deploys the system.

| Review | Scope | Full report |
|---|---|---|
| Core-code audit | perception, XAI, fusion, pipeline, I/O; every finding reproduced with a script | [research/audit_core.md](research/audit_core.md) |
| Security audit | API, auth, RBAC, storage, MQTT, cameras, CI; live probes (uvicorn, Playwright, clean venvs) | [research/audit_security.md](research/audit_security.md) |
| Evaluation audit | leakage, metric correctness, every claim in RESULTS, confidence intervals | [research/audit_eval.md](research/audit_eval.md) |
| Standards gap analysis | IEC 62676, FM 3232 / NFPA 72 / IS 2189, ISO/IEC 42001 / 23894 / 25010, NIST AI RMF, OWASP ASVS / API Top 10, IEC 62443, DPDP Act 2023 + Rules 2025, EU AI Act, SBOM / SLSA, SRE | [research/standards_gap.md](research/standards_gap.md) |

## 1. What the audits found, and what happened to each finding

### Code defects: 20 found, 20 fixed, each with a regression test

`tests/test_core_regressions.py` has a test named after each finding (`test_c1_…` to `test_c20_…`).
The ones that mattered most:

| id | Finding | Fix |
|---|---|---|
| C1 / C2 | **Privacy leak.** Faces were not blurred on fire-only cameras (the person model didn't run), and newly seen low-confidence people were never blurred | Person detection always runs when privacy ≠ raw. The filter uses *all* detections, and blurs the whole frame if no person model is loaded (fail closed) |
| C3 | A missing sensor silently removed the **"Trigger evacuation"** step (`prob > 0.85 or smoke_ppm > 600`) | Lazy boolean evaluation, where a missing name makes only its own comparison false |
| C4 / C5 | **Slow smoke was invisible.** The smoke baseline adapted in about 3 s, and the fire detector ran once in 900 static frames while serving stale detections | Time-based asymmetric baseline (15 min up / 20 s down). The fire detector runs on its own clock at least every second, and detections expire |
| C6 | An alert could fire while its own text said "Confidence 4 %" (decided on the smoothed score, explained with the raw one) | Onset needs the raw probability ≥ threshold, and the smoothed state is stored on the assessment |
| C7 | The grounding verifier accepted swapped values, wrong hazard words and false causes | Values are bound to their cues, every hazard word is checked, causal claims must name a cue that pushed toward the alarm, and the playbook strings are cut out exactly, so nothing can hide after "Recommended:". Detection went from 63 % to **99.8 %**, with 0.5 % false rejects |
| C8 | Two isolated one-frame spikes minutes apart caused an alarm | Hysteresis counts only consecutive frames |
| C11 / S1 | One exception killed a camera thread while health kept saying "live" | Each step is guarded, raises a `processing_error` trouble alert, and the watchdog checks thread liveness |

### Security findings: 13 found, 13 fixed, each with a regression test

`tests/test_security_regressions.py` (`test_s1_…` to `test_s13_…`). The critical and high ones:

| id | Finding | Fix |
|---|---|---|
| S1 | A race in the sensor hub crashed camera threads while `/healthz` stayed green | Thread-safe hub. `/healthz` returns 503 when a camera thread is dead or stale |
| S2 | Any device key could inject or suppress smoke readings in **any** zone | Device keys are bound to zones (403 otherwise). On MQTT the topic, not the payload, decides the zone |
| S3 | A 300 MB chunked upload was buffered before auth (container OOM) | Pure-ASGI body limit that stops at 64 KB, before auth |
| S4 | RTSP passwords leaked into the viewer UI, the database and Slack/Teams | URL credentials are redacted everywhere |
| S5 | Re-dismissing one alert 400 times silenced fire detection permanently (threshold 0.20 → 0.995) | One label per alert (409 after that). Calibrated alert lines are clipped to [`min_threshold`, `max_threshold` = 0.9] |
| S6 | A torn calibration file crash-looped the service | Atomic write (temp file + fsync + replace). A corrupt file loads as uncalibrated |

The medium and low findings were also fixed: zone allow-list and caps (S7), configurable
trusted proxies and bounded lockout (S8), CI type errors (S9), the NaN echo that caused a 500 (S10),
the WebSocket Origin check (S11), the HMAC-keyed audit chain with a head anchor (S12), and a
private OpenAPI schema plus a TLS proxy in compose (S13).

### Evaluation findings: what was corrected

| Finding | Action |
|---|---|
| **Fire test-set leakage.** 58 % of Home-fire test images were near-duplicates of training images, and DBA was split by file name, so video frames fell on both sides | New leakage-free protocol (`scripts/prepare_fire_data.py` v2): pHash scene grouping, scene-level split, and removal of every training image within 10 bits of any test image (2,211 removed). Retrained and re-evaluated. See RESULTS §4 |
| A plan criterion (fire recall ≥ 0.85 at ≤ 10 % FPR) was missing from the table | Now reported, with its status |
| Fall, crowd and fire-video benchmarks were also used during development | Stated in every section. A frozen-system **external sanity check** on footage never used in development was added (RESULTS §8) |
| No confidence intervals | 95 % intervals on every headline number |
| Small number errors (99.2 % → 99.1 %, 306 → 305 counterfactuals, descent-speed ranges, GBM comparison, an unverifiable 0.64 → 0.81 claim) | Corrected or removed |
| Saliency was described as better than the controls, but it beats only random | Reported as failing the centre-control criterion. The heat map is presented as a visual aid; the evidence ledger is the explanation |

### Second round (24 Oct): new benchmarks and an end-to-end run found four more real problems

| Found by | Problem | Fix |
|---|---|---|
| Privacy verification with an independent face detector (`eval_privacy.py`) | The Gaussian head blur still let a face detector find 26 of 184 faces (14 %) | Coarse head pixelation: 0 / 184 |
| External check on walking footage | Fall weights fitted on GMDCSA24 learned a single-cue shortcut (descent speed) and raised 17 false fall alarms in 7 minutes | Fitted profile rejected; the config now enforces a corroboration rule (`min_cues: 2`: no single cue can raise a fall) and the fall alert line can't be calibrated below the point where one cue would suffice |
| Running the console end to end | SIGTERM hung forever once frames stopped (idle WebSocket handlers); a fall re-acquired under a new track id paged twice | WebSocket heartbeat + 5 s graceful-shutdown bound; incidents re-identified by place |
| UR-Fall input check | Depth and RGB side by side fed to the models in the first external run | RGB half only; numbers corrected and disclosed |
| Webcam field test by the team | Lying down slowly on a bed pushed the fall score over the line: posture cues alone could reach it (6 of 7 GMDCSA24 false alarms were lie-downs) | Fall evidence gate: an uncontrolled descent (new hip-drop cue) *and* a lying end posture are both required, as exact named ledger terms shown on the gauge; lie-down false alarms 6 → 0, GMDCSA24 F1 0.814 → 0.883, UR-Fall 19 → 23 / 30 (RESULTS §2.1) |
| Second webcam test | Fall ALARMs at start-up from tracks never seen upright (hip drop measured against a horizontal box); a quiet lie-down left no record to check | Hip-drop cue gets the same reliability discount as descent speed; every gate hold is written to the audit trail and the export (`hazard_held`) |
| Third webcam test | Lying down on the floor from standing in 2-3 s still alarmed: hip drop measured distance over 3 s, not speed | Fastest 1-s hip drop (looked up over the transition, same memory as descent speed); a first attempt that cost falls on every benchmark was rejected; final version loses nothing on any benchmark (RESULTS §2.1) |

Added in the same round: a second, independent fall dataset (UR-Fall: 24 / 30 with the fall
profile after the evidence gate, 0 / 5 alarms on daily activities); per-camera pose models; long-lie escalation;
cross-camera incident grouping; shift-handover report; restricted-area intrusion rules;
an exact What-if explorer; an hourly end-to-end self-test with trouble alerts; calibration (ECE)
and corruption-robustness benchmarks; and a log of the ideas that did not work (GAM shape
functions, key-point smoothing, constrained re-fit, hard-negative fine-tune), so that nobody
repeats them.

## 2. Standards position

The standards analysis separates three bars that "production-ready" can mean.

**1. Software and security engineering: met for a pilot.**
Authenticated RBAC API (scrypt, pinned HS256 JWT, revocation, lockout); device keys bound to zones;
rate and body limits; CSP and security headers; tamper-evident HMAC audit chain covering logins,
acknowledgements, exports, pauses, mutes, erasure and legal hold; encrypted media with retention purge,
legal hold and erasure; trouble signals for offline, frozen, blackout, defocus, view-change and
processing errors, routed through the same escalation chain as alarms; an audible console alarm with
distinguishable alarm, verify and trouble tones; systemd watchdog; Prometheus metrics with Alertmanager
and SLO burn-rate rules; Caddy TLS and MQTT over TLS in compose; strict config schema; SHA-256 pinned
models (the service refuses a mismatch); CycloneDX SBOM + ML-BOM; lockfile; CI with ruff, mypy,
bandit, pip-audit, tests with a coverage gate, and SLSA build provenance.

What is still missing at this bar needs hardware or time rather than code: a 24 h soak test on the
target Pi 5 / Jetson, a penetration test, and signed container images in a registry.

**2. AI governance and privacy: documents drafted, sign-off needed.**
The repository has a risk register (FMEA), an AI governance note mapped to ISO/IEC 42001 / NIST AI RMF,
a DPIA-style privacy document for the DPDP Act and Rules, model cards, intended and prohibited uses,
and post-deployment monitoring (per-camera dismissal rate and calibration drift metrics). These are
drafts. ISO 42001 and the DPDP Act need an organisation to own them: a DPO who signs the DPIA and
chooses the lawful basis, signage and a grievance officer, a breach procedure (DPB without delay plus
72 h; CERT-In 6 h), and operator training on automation bias.

**3. Life-safety certification: not applicable, by design.**
No student or open-source project can be a listed fire alarm (IS 2189 / NBC 2016, NFPA 72,
UL 268B / FM 3232, EN 54). The system is therefore positioned, in the README, console footer, every
fire alert, every notification and every incident report, as **decision support that supplements and
never replaces the certified fire alarm system**. It never dispatches anyone automatically, which
also keeps it outside the EU AI Act's high-risk "emergency dispatch" category (Annex III 5(d)).

## 3. Verdict

**Engineering: industry standard for a pilot.** The code, security, operations and supply-chain
practices match what a professional team would ship to a supervised campus pilot. Every defect the
independent audits found is fixed and pinned by a test (130 tests). The explainability core, an
exact additive evidence ledger with counterfactuals and a grounding verifier, is stronger than what
commercial video-analytics products expose.

**Accuracy: honest, not yet field-proven.** Falls: F1 0.81–0.84 on the development set (target
0.90, not met), but on a second dataset never used in development the fall profile caught
23 / 30 UR-Fall falls and 9 / 11 external falls with no false fall alarm on ordinary walking
footage. Fire: the leakage-free image detector scores mAP 0.56–0.65 and finds only two thirds of
fire images at a 10 % false-positive budget (the plan asked for 85 %); the fused fire branch,
which needs corroboration, still caught 15 / 17 FURG fire videos with no false alarm. Crowd and
fire-video results come from development sets with wide intervals, and the one false alarm on
unseen footage came from an uncalibrated camera. That is why the system triages alerts into
ALARM and VERIFY, calibrates each camera, refuses single-cue fall models, and keeps a human in
the loop. The number that matters, false alarms per camera-day and misses on *your* campus
cameras, can only come from a field trial.

**Not production for life safety, and it must never be presented as such.** Before a real
deployment the institution must: run a supervised field trial on the real cameras with a
ground-truth log; complete DPO sign-off and signage under the DPDP Act; commission a penetration
test; train operators; consult the fire officer; and resolve licensing (the Ultralytics weights are
AGPL-3.0 and the Home-fire data is CC BY-NC 4.0, which is acceptable for a non-commercial academic
pilot but blocks commercial use without retraining or a commercial licence).
