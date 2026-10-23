# Standards and production-readiness gap analysis
## Explainable Edge AI Copilot for Smart Emergency Response

Prepared 2026-10-23. Status checks are against the repo at commit `ba0be72` ("Architecture and privacy (DPIA summary) docs"). That includes the production layer added today (`ebb31d1`, `4e0b323`, `374a08c`, `87712db`): auth/RBAC, SQLite store, hash-chained audit log, encrypted media, retention, notifications, multi-camera workers, Prometheus, compose stack and docs. Not legal advice. Where a source could not be read in full, this says so.

---

## 0. Bottom line

* **"Production-ready" for this system means three different bars:**
  1. **Software/security engineering.** ISO/IEC 25010, OWASP ASVS / API Top 10, IEC 62443, SRE, supply chain. Students can meet most of this in code.
  2. **AI governance and privacy.** ISO/IEC 42001 / 23894, NIST AI RMF, DPDP Act + Rules, EU AI Act if exported. These need documents that a real organisation signs off: DPO, safety committee, campus IT.
  3. **Life-safety certification.** IS 2189 / NBC 2016, NFPA 72, UL 268B / FM 3232 / EN 54. A student system **cannot** meet this, and it **must not be positioned as a fire alarm**.
* **Where the repo stands.** After today's commits it is a **credible pilot-grade decision-support system**: authenticated RBAC API, tamper-evident audit, encrypted and retained media, escalation, camera health, validated config, CI quality gates. The remaining gaps that matter most are:
  1. "Who watches the watcher": camera faults and Prometheus alerts don't reach a human, there is no software watchdog, and no Alertmanager.
  2. No operator stop/pause control, erasure endpoint or legal hold.
  3. No supply-chain evidence: SBOM, lockfile, signed images or model manifest.
  4. No NTP/UTC discipline.
  5. No field validation on campus footage, and no risk register.
  6. The fire-alarm disclaimer is only in OPERATIONS.md and the model card. It is missing from the README and the console.

---

## 1. Fire detection: can a video/AI detector replace certified fire alarms?

**No. Position it as supplementary early-warning and decision support.** The evidence:

| Source | What it says |
|---|---|
| **IS 2189:2008** (Indian code of practice for automatic fire detection and alarm) | Covers heat, smoke (ionisation/optical), aspirating, spark/ember, UV/IR flame and multi-sensor detectors. **Video detectors are not mentioned.** It points to IS 11360 (smoke) and IS 2175 (heat). "Approved" means acceptable to the authority having jurisdiction. It sets daily, weekly, quarterly and annual test and maintenance duties, fault signalling and standby power. |
| **NBC 2016 Part 4** | Requires automatic detection and alarm per IS 2189 (addressable smoke detectors + manual call points) for many occupancies. The fire alarm and PA run on emergency power. The Chief Fire Officer issues the fire NOC. Group B (educational) buildings 9 m or taller, or with more than 500 m² on any floor, are in scope. |
| **NFPA 72** (US) | Recognises video image smoke detection (VISD) and flame detection (VIFD). They must be **listed**. Software and controls must be protected from unauthorised change, and changes re-tested under Ch. 14. Video may go to other systems only via manufacturer-provided outputs. |
| **NYC Buildings Bulletin 2014-005** | Accepted VISD only as a listed alternative in large open spaces. Approved fire alarm systems were still required. VISD may not be used for heat detection. The listed product gives dry-contact alarm and trouble signals to the fire alarm control panel. (Bulletin later rescinded prospectively.) |
| **FM Approvals Class 3232 (2023)**, video image fire detectors | Smoke tests based on UL 268, plus fire tests at minimum and maximum distance. Nuisance immunity (sunlight, welding, lamps). Environmental and EMC tests. **Alarm to the fire panel within 30 s.** **Trouble signals** for loss of camera power, video path, image contrast or image clarity, and for software failure. Power cycling must not cause an alarm. |
| **CAN/ULC-S592** (Canadian VISD standard, built on ULC-S529) | VISD devices "are not intended as direct replacements" for standard open-area detectors. |
| **UL 268B** | The UL VISD standard. Its text was not accessible during this research, so its specific requirements are not verified here. |
| **ISO/TS 7240-29:2017** (VFD product spec, now withdrawn) / **ISO/TS 7240-30:2022** (design, install, service) / **prEN 54-34** (draft EN for video fire detectors, project started 2026) | Video fire detection is only now entering EN 54. The FIA notes that even ISO/TS 7240-30 requires independent certification, at least 24 h standby + 30 min alarm power, camera-alignment checks, and adequate illumination where video is the sole protection. |
| **EU AI Act / Machinery** | Not relevant to the fire-alarm question. |

**What the repo must do:**
1. State "not a certified fire alarm; the building's IS 2189 system remains the life-safety system" in the README, console footer, every fire alert text and the incident report. Today it is only in `docs/OPERATIONS.md` and `models/README.md`.
2. Never auto-trigger evacuation or call 101. Keep the human in the loop: the playbook already says "Trigger evacuation…" as a recommendation.
3. Never wire it as an initiating device on the fire panel. At most, after the campus fire contractor and CFO agree, use an auxiliary/supervisory input.
4. Borrow FM 3232's engineering ideas anyway: a trouble state for video loss, contrast or clarity loss and software failure; a 30 s alert latency budget; nuisance-source testing (sunlight, welding, orange lamps, steam). These are measurable and cheap.

---

## 2. Requirements by standard / practice (what a production system is expected to provide)

### 2.1 IEC 62676 (video surveillance systems; adopted in India as IS 16910, e.g. IS 16910 (Part 1/Sec 2):2018 = IEC 62676-1-2)
Grades 1–4; a campus is typically grade 2–3. Checkable requirements, taken from Bosch's IEC 62676-1-1 conformance summary:
- **Tamper detection and alarm:** video loss, video substitution, contrast reduction, field-of-view change, obscuring/blinding. Tamper must be logged distinctly from faults and alarms.
- **Time:** components within ±10 s of UTC at grades 3–4. Records labelled in UTC with a local offset; one-second time resolution.
- **Logs:** chronological, timestamped, protected from edit or delete. They must include logins (including denied ones with the reason), configuration and time changes, exports, acknowledgements, start/stop and failures. Log deletion is not allowed at any grade.
- **Users:** authorisation scheme, no clear-text passwords. Exporting and deleting data are reserved for higher roles.
- **Export integrity:** authenticatable at grades 3–4. Exports carry source ID and timestamps.
- **Storage:** configurable maximum retention with automatic deletion; **protected recordings exempt**. Warn when capacity is low.
- **Alarm handling:** FIFO unless prioritised; indicate when more alarms exist than are shown; visual + audible; acknowledgeable. Alarm, fault and tamper are distinguishable.
- **Failure notification:** essential-function failure indicated within 100 s (grade 3–4). Interconnection checks about every 30 s, with operator notification within 180 s at grade 3.

### 2.2 ISO/IEC 42001:2023 (AI management system), Annex A controls
A.2 AI policy · A.3 roles, concern reporting · A.4 resources (data, tooling, compute, people) · **A.5 AI impact assessment** (individuals, groups, society) · **A.6 lifecycle**: requirements, design documentation, **verification and validation**, **deployment with release criteria and rollback**, **operation and monitoring (drift, failures, misuse)**, technical documentation, **event logs** · A.7 data provenance, quality, preparation · **A.8 user information, external reporting channel, incident communication** · **A.9 responsible use, human oversight, intended use (no function creep)** · A.10 suppliers (pre-trained models). Certification is organisational. The repo can supply the A.6/A.7/A.8 artefacts.

### 2.3 ISO/IEC 23894:2023 (AI risk management, used with ISO 31000)
Identify risks against AI objectives: safety, robustness, transparency, privacy, fairness, maintainability, data quality. Use AI risk sources: environmental complexity, level of automation, ML-specific issues, hardware, lifecycle, technology readiness. Then analyse, evaluate, treat, monitor and record. **Deliverable:** a risk register / FMEA for the system's hazards, with owners and treatments.

### 2.4 NIST AI RMF 1.0
GOVERN (policies, roles) · MAP (context, intended use, limits) · MEASURE (TEVV, safety, robustness, privacy) · MANAGE. Key checkable outcomes:
- **MANAGE 2.4:** mechanisms to supersede, disengage or deactivate the system.
- **MANAGE 3.2:** monitor pre-trained models (YOLO weights).
- **MANAGE 4.1:** post-deployment monitoring plan (user input, override, incident response, change management).
- **MANAGE 4.3:** incident tracking and communication.
- **MANAGE 1.4:** document residual risk for users.

### 2.5 ISO/IEC 25010:2023 (product quality)
Nine characteristics. The ones that bite here:
- **Safety** (new in 2023): operational constraints, risk identification, **fail safe**, **hazard warning**, safe integration.
- **Reliability:** faultlessness, fault tolerance, availability, recoverability.
- **Security:** confidentiality, integrity, non-repudiation, **accountability**, authenticity, resistance.
- **Interaction capability:** user-error protection, self-descriptiveness.
- **Flexibility:** installability, replaceability.

Each needs a measurable target, for example availability ≥ 99.5 % per camera.

### 2.6 Security: IEC 62443, OWASP ASVS 5.0, OWASP API Security Top 10 (2023)
- **IEC 62443-4-2 foundational requirements:** FR1 identification and authentication · FR2 use control (RBAC, auditable events, timestamps, non-repudiation) · FR3 system integrity (software/firmware integrity, input validation, audit protection) · FR4 data confidentiality (crypto, erase) · FR5 restricted data flow (zones and conduits, i.e. a camera VLAN) · FR6 timely response (audit access, continuous monitoring) · FR7 resource availability (DoS, backup/recovery, least functionality, component inventory). A pilot should target roughly SL1–SL2.
- **OWASP ASVS 5.0 (2025):** L1 is the floor and L2 the norm for a system holding personal data. Includes V6 authentication (passwords ≥ 8, recommended ≥ 15), V11 crypto and V16 logging/error handling (log security-control failures). ASVS 5.0 contains no AI controls; those sit in the separate AISVS project.
- **OWASP API Top 10 (2023):**
  - API1/3/5: object, property and function-level authorisation.
  - API2: authentication.
  - API4: resource consumption.
  - API7: misconfiguration (default credentials, CORS, TLS).
  - API8: automated threats.
  - API9: inventory.
  - API10: unsafe consumption, which here means sensor, MQTT and webhook payloads.

### 2.7 ONVIF
- Profile S/T: streaming and events, RTSP. **Profile M: analytics metadata and events.** Metadata stream mandatory; MQTT/JSON event delivery conditional; object classification.
- A third-party analytics server that wants to interoperate should:
  - discover and configure cameras via ONVIF (WS-Discovery, GetStreamUri), not hand-typed RTSP URLs;
  - publish its own events in a VMS-consumable form (ONVIF events or Profile-M-style MQTT JSON) so Milestone, Bosch and similar VMS products can ingest alerts.
- **India CCTV Essential Requirements.** From 1 April 2026, CCTV cameras sold in India must meet the ERs and hold STQC/BIS certification: no hard-coded credentials, secure firmware/update, encrypted communication, tamper protection. The campus should buy only ER-certified cameras.

### 2.8 Fire standards (NFPA 72 / UL 268(B) / FM 3232 / EN 54 / IS 2189 / NBC 2016)
See §1. For this repo: a supplementary disclaimer everywhere, no automatic actuation, FM-3232-style trouble supervision, a latency budget of 30 s or less, and nuisance-immunity testing.

### 2.9 India DPDP Act 2023 + DPDP Rules 2025
- **Timeline.** Rules notified 13 Nov 2025. Rule 3 (notice), Rules 5–16 (including security safeguards, breach notification and retention) and Rules 22–23 come into force **18 months later, i.e. 13 May 2027**. Consent Manager provisions apply from 13 Nov 2026.
- **Lawful basis.** Consent with a plain-language, itemised notice, or a **Section 7 legitimate use**. Relevant ones:
  - (f) medical emergency threatening life or health;
  - (h) safety or assistance during a disaster or breakdown of public order;
  - (i) employment purposes, for staff.

  Whether everyday campus fall or crowd monitoring fits (f)/(h) rather than needing notice is for the institution's DPO to decide. Notices and signage are needed either way.
- **Rule 6, reasonable security safeguards:**
  - (a) encryption, obfuscation, masking or tokens;
  - (b) access control;
  - (c) access logs and monitoring;
  - (d) backups / continuity;
  - (e) **keep logs and personal data for one year** for investigation unless another law says otherwise;
  - (f) processor contracts;
  - (g) organisational measures.
- **Rule 7, breach.** Inform affected persons and the Data Protection Board **without delay**, then send the Board a detailed report **within 72 h**.
- **Rights.** Access, correction, erasure, grievance (redress within 90 days), nomination. Erase once the purpose is served.
- **Children.** Section 9(1)/(3) (verifiable parental consent; no tracking or behavioural monitoring of children). The Fourth Schedule Part A exempts educational institutions for tracking/behavioural monitoring done for educational activities or **in the interests of the safety of enrolled children**. This matters if under-18 students are on campus.
- **Related Indian obligations.**
  - **CERT-In Directions (28 Apr 2022):** report cyber incidents within **6 h**; keep logs for **180 days in India**; sync clocks to NIC (samay1/2.nic.in) or NPL (time.nplindia.org) NTP; designate a point of contact. These apply to body corporates and government organisations; the university decides applicability.
  - **MeitY India AI Governance Guidelines (Nov 2025):** non-binding; seven "sutras"; recommend AI incident reporting.

### 2.10 EU AI Act (only if exported or deployed in the EU)
- **Not remote biometric identification.** No face recognition and no matching against a database, so it is not Annex III 1(a) and not the Art. 5(1)(h) real-time RBI ban. Keep it that way: never add face ID or re-ID.
- **Emotion recognition (Art. 5(1)(f) ban in education and workplaces).** Recital 18 excludes physical states. The Commission guidelines put crowd-control systems outside the biometric-based ban. However, "crowd **panic**" framing and anything that infers fear or aggression would invite scrutiny. Name the hazard *abnormal crowd movement* and infer only kinematics.
- **Annex III 5(d).** Systems that **dispatch or prioritise** emergency first responders, or do emergency triage, are high-risk. Recommending "call 108" and assigning severity is close to this. Keeping a human decision between alert and dispatch, with no auto-dispatch, is the main mitigation. If the system were treated as high-risk, it would need Art. 9 risk management, Art. 10 data governance, Art. 11 technical documentation, **Art. 12 automatic logging over the lifetime**, Art. 13 instructions for use, **Art. 14 human oversight** (understand limits, beware automation bias, override, **stop button**), Art. 15 accuracy/robustness/cybersecurity, plus a conformity assessment.
- **Dates.** The AI Omnibus, Regulation (EU) 2026/1744 (OJ 24 Jul 2026, in force 27 Jul 2026), moved Annex III obligations to **2 Dec 2027** and Annex I (product) obligations to **2 Aug 2028**. The Art. 5 prohibitions have applied since Feb 2025.

### 2.11 MLOps maturity (Google)
- **Level 0:** manual and script-driven, with no production monitoring.
- **Level 1:** an automated training pipeline (continuous training) with data and model validation, a metadata store and triggers.
- **Level 2:** CI/CD of the pipeline, a model registry and staged rollout.
- **Realistic target for a pilot:** "Level 0.5". Reproducible scripts (already present), a model registry/manifest with hashes and metrics, a release gate that re-runs the benchmark suite, drift monitoring and rollback.

### 2.12 Model cards / datasheets / ML-BOM
- **Model card (Mitchell et al.):** details/version/licence, intended and **out-of-scope** use, **factors** (camera hardware, lighting, environment, groups), metrics and thresholds, evaluation data, training data, **disaggregated results** with confidence intervals, ethical considerations, caveats.
- **Datasheets (Gebru et al.):** provenance, collection, licence and consent for each dataset.
- **CycloneDX ≥ 1.5 ML-BOM:** `machine-learning-model` components with `modelCard`, datasets as data components, and pedigree (base model, then fine-tuned, then INT8).

### 2.13 SBOM and supply chain (CycloneDX/SPDX, SLSA, signing)
- An SBOM per release covering Python deps, the OS packages in the image and the models.
- **SLSA v1.0 build levels:**
  - L1: provenance exists.
  - L2: hosted build with signed provenance, e.g. GitHub artifact attestations.
  - L3: hardened build.
- **Model signing:** OpenSSF `model-signing` v1.0 (Sigstore, PKI or self-signed) to verify model files before loading.
- Pinned, hash-locked dependencies and a pinned base-image digest.

### 2.14 Observability and SRE (OpenTelemetry, Prometheus, SLOs, runbooks)
- **Prometheus conventions:** app prefix, base units (`_seconds`), `_total` for counters, low-cardinality labels; histograms for latency.
- **OpenTelemetry Python:** traces and metrics stable, logs still in development. Tracing the alert path (frame, then alert, then stored, then notified, then acknowledged) is the useful case.
- **SRE workbook:** SLIs as good/total events; SLOs with error budgets; **multi-window multi-burn-rate alerts** (page at 2 % of budget in 1 h or 5 % in 6 h; ticket at 10 % in 3 d). Low-traffic services need synthetic probes, which here means a periodic **synthetic test alert per camera** end-to-end.
- **Runbooks** per alert, plus post-incident review.

### 2.15 Twelve-factor app
Codebase · explicit dependencies · **config in the environment** (secrets at least) · backing services · build/release/run separation · stateless processes · port binding · concurrency · **disposability** (fast start, graceful shutdown) · dev/prod parity · **logs as event streams** · admin one-off processes.

---

## 3. Compliance matrix

Status key: **met** = implemented and in code or docs at `ba0be72`; **partial** = some of it; **missing** = absent. "Org" in the Action column means it needs the institution, not just code.

| Standard / practice | Requirement | Status in repo (evidence) | Action |
|---|---|---|---|
| IS 2189 / NBC 2016 / NFPA 72 / FM 3232 / EN 54 | Not presented as, or wired as, a fire alarm | **partial**: `docs/OPERATIONS.md` "Limits" and `models/README.md` say "not a certified fire alarm"; README, console and alert text don't | Add the disclaimer to README, console footer, fire alert text, incident report and notifications (30 min) |
| FM 3232 §4.5 / NFPA 72 trouble | Distinct trouble signal for video loss, contrast/clarity loss, software failure | **partial**: `cameras.py` TamperDetector (frozen/blackout/defocus) + offline state, audited and broadcast (`service._on_fault`); faults are **not** sent through `Notifier`; no software-failure watchdog (`WatchdogSec=0` in systemd unit) | Route `camera_fault` and pipeline exceptions to the escalation chain; add sd_notify watchdog heartbeat from each camera loop |
| FM 3232 §4.3 | Alarm within ≤ 30 s of detection | **partial**: decision 30–140 ms (RESULTS §6); end-to-end time to notification not measured | Measure and export detection-to-notification latency; add synthetic end-to-end probe |
| FM 3232 §4.4 | Nuisance immunity testing | **partial**: neutral-image FPR 18.5 % (detector), FURG 1 FA / 4.7 min; no sunlight/welding/orange-lamp/steam campus tests | Run a nuisance test set on campus (canteen steam, sodium lamps, welding lab) |
| IEC 62676 tamper | Video loss, substitution, contrast, FOV change, obscuring | **partial**: frozen/blackout/defocus/offline in `cameras.py`; FOV change is detected as `scene_change` in the pipeline but **not raised as a fault**; no substitution detection | Raise `scene_change` as a `fov_changed` fault; add reference-frame similarity check against commissioning snapshot |
| IEC 62676 time / CERT-In | UTC within ±10 s; NTP to NIC/NPL; UTC + offset in records | **missing**: `wall_time` is epoch; report uses `time.localtime`; no NTP guidance or clock-skew metric | ISO-8601 UTC with offset in DB, logs and reports; chrony config (samay1.nic.in, time.nplindia.org) in OPERATIONS.md; skew gauge |
| IEC 62676 / 62443 FR2 / DPDP Rule 6(c) | Tamper-evident audit log of logins (incl. failed), acks, exports, config, start/stop | **met**: `storage.py` hash-chained audit, `verify_audit_chain`, `/api/audit/verify`; login, login_failed, clip_viewed, report_exported, alert_confirmed/dismissed, service_start/stop, camera_fault, retention_purge | Add config-change and model-change events (on start, diff the provenance against the last run); export audit to off-box storage |
| IEC 62676 storage | Max retention with auto-delete; **protected recordings exempt**; capacity warning | **partial**: `Store.purge` hourly, `retention_days` (default 30); no legal-hold flag; no disk-capacity metric | Add `protected` flag (admin, audited) excluded from purge; disk free gauge + alert |
| IEC 62676 alarm handling | Visual + audible, acknowledgeable, alarm/fault/tamper distinguishable | **partial**: console shows alerts and faults, confirm/dismiss; no audible alarm found in `static/app.js` | Add audible cue for ALARM-mode alerts and faults (with mute logged) |
| ONVIF Profile S/T/M | Discovery, stream URI, event/metadata interop | **missing**: RTSP via OpenCV only; MQTT alert publish exists but no ONVIF | Add ONVIF discovery + GetStreamUri (onvif-zeep) for commissioning; document MQTT JSON alert schema; P2 |
| India CCTV ER / STQC | Certified cameras | **n/a (org)** | Procurement note in OPERATIONS.md: only ER/STQC-certified cameras; no default credentials; camera VLAN |
| ISO/IEC 42001 A.2/A.3 | AI policy, roles, concern reporting | **missing (org)** | Draft a one-page AI use policy + RACI (operator, admin, model owner, DPO) for the pilot |
| ISO/IEC 42001 A.5 | AI impact assessment | **partial**: `docs/PRIVACY.md` (DPIA summary: data map, lawful basis, rights, privacy by design) | Extend to impacts of false negatives/positives, automation bias and groups (people with disabilities, children); DPO sign-off (org) |
| ISO/IEC 42001 A.6.2.4 / NIST MEASURE | Verification and validation for the intended context | **partial**: strong public-dataset benchmarks (RESULTS.md); **no validation on campus cameras** | Campus field trial with ground-truth log (weeks; org) |
| ISO/IEC 42001 A.6.2.5 / MLOps | Release criteria + rollback | **partial**: acceptance gate in PLAN.md; provenance (version, config hash, model sha256) per alert (`service.py`); no automated release gate, no rollback procedure | CI job that re-runs XAI/unit benchmarks on fixtures; tagged images + documented rollback |
| ISO/IEC 42001 A.6.2.6 / NIST MANAGE 4.1 / AI Act Art. 72 | Post-deployment monitoring (drift, misuse) | **partial**: `edgecopilot_operator_feedback_total{outcome}` (no camera label), bounded online recalibration; no drift metrics/alerts | Per-camera dismissal rate, alerts/hour, VERIFY ratio, calibration n, as metrics + alert rules; monthly review |
| ISO/IEC 42001 A.6.2.8 / AI Act Art. 12 | Event logs over the lifetime | **met**: SQLite alerts + audit + JSON logs (`logging_setup.py`) | Define log retention ≥ 1 yr (DPDP Rule 6(e)) / 180 d (CERT-In) separate from media retention |
| ISO/IEC 42001 A.7 / datasheets | Data provenance, licences | **partial**: README licence list, model card start points | Datasheet per dataset; flag **CC BY-NC (Home-fire)** and **AGPL-3.0 (Ultralytics)** as blockers for any commercial use |
| ISO/IEC 42001 A.8.3/A.8.4 / NIST MANAGE 4.3 | External reporting channel; incident communication | **partial**: SECURITY.md vulnerability contact; incident report export (`/report.html`) | AI incident/near-miss register + post-incident review template; grievance contact on signage (org) |
| ISO/IEC 42001 A.9.4 | Intended use, no function creep | **partial**: PRIVACY.md forbids discipline/attendance use | Put "intended / prohibited uses" in README and console About; config guard that `privacy: raw` needs admin + is audited |
| ISO/IEC 23894 / NIST MAP | Risk register / hazard analysis | **missing** | FMEA: missed fire, missed fall, false evacuation, camera down, sensor spoofing, model tampering, operator overload; owners + treatments (team can draft in 2–3 h) |
| NIST MANAGE 2.4 / AI Act Art. 14(4)(e) | Stop, disengage, override | **partial**: override via confirm/dismiss; VERIFY routing; camera `enabled` only in config (restart); no runtime pause/stop | Admin-only `POST /api/cameras/{id}/pause|resume` + global maintenance mode, audited, with a visible "NOT MONITORING" banner and metric |
| AI Act Art. 14(4)(b) | Automation-bias awareness | **partial**: OPERATIONS.md "VERIFY means look first" | Operator training note + periodic dismissed/confirmed review (org) |
| ISO/IEC 25010 Safety: fail safe / hazard warning | Degraded state is loud, not silent | **partial**: `/readyz` 503 when no camera live; sensor staleness lowers reliability; Prometheus alert rules exist but **no Alertmanager**, so rules notify nobody | Add Alertmanager to compose → same webhook/email; console "system degraded" banner |
| ISO/IEC 25010 Reliability | Reconnect, recoverability | **met**: exponential back-off reconnect (1→30 s), systemd `Restart=always`, compose `restart: unless-stopped`, SQLite WAL | Soak test 24 h with network pulls (needs hardware) |
| ISO/IEC 25010 Reliability: backups | Backup + tested restore | **partial**: SECURITY.md checklist item only | `scripts/backup.sh` using SQLite online backup + key escrow; restore drill |
| ISO/IEC 25010 Flexibility: install/update | Versioned OTA with rollback | **partial**: Docker image tag 1.1.0, models mounted read-only; no OTA/rollback tooling | For pilot: pinned image digests + `rollback.sh`; fleet OTA (Mender/RAUC) only with real fleet (org) |
| OWASP API2 / IEC 62443 FR1 / ASVS V6 | Authentication | **met**: scrypt hashes, min 10 chars, JWT HS256 pinned, `exp/sub/jti` required, revocation, lockout, HttpOnly SameSite=Strict cookie (`security.py`, `server/app.py`) | Optional TOTP for admins; password change flow |
| OWASP API1/3/5 / FR2 | Function/object authorisation | **met**: viewer < operator < admin deps; clip/report = operator; audit = admin | Single-tenant, so BOLA risk is low; add per-zone scoping if several departments share a box |
| OWASP API4 / FR7 | Resource limits | **met**: rate limiter, 64 KB body limit, WS client cap + bounded queues, frame dropping | — |
| OWASP API7 / ASVS V12 / FR4 | TLS everywhere, secure defaults | **partial**: CSP and security headers, refuses non-loopback without auth, Secure cookie behind HTTPS; **TLS not shipped** (docs only); MQTT pilot listener 1883 plain; console loads Google Fonts from the internet | Ship Caddy/nginx with campus CA in compose; MQTT 8883 + per-device passwords; self-host fonts (edge boxes may be offline; avoids third-party requests) |
| OWASP API10 | Validate external inputs | **met**: pydantic `SensorReading` ranges, MQTT payload validation, server-side timestamps, device API keys | ESP32 sketch still posts without `X-Api-Key` and over HTTP: update the sketch |
| IEC 62443 FR3 / NFPA 72 17.7.7.4 | Software/model integrity; protect against unauthorised change | **partial**: model sha256 recorded in provenance; config validated at start; nothing **refuses** an unexpected model | Model manifest (sha256 allowlist) checked at load, refusing on mismatch; optional OpenSSF model-signing |
| IEC 62443 FR5 | Network segmentation | **partial (org)**: SECURITY.md VLAN checklist | Campus IT: camera VLAN, egress allowlist |
| ASVS V16 / 12-factor XI | Structured logs to stdout | **met**: `logging_setup.py` JSON formatter | Include UTC ISO time and request ID |
| Config validation | Fail fast on bad config | **met**: `config_schema.py` (strict pydantic, cross-checks, CLI lint in CI) | Add schema version field + migration note |
| 12-factor III | Secrets/config from environment | **met** for secrets (`.env.example`, env keys); config YAML mounted read-only | — |
| 12-factor IX | Disposability, graceful shutdown | **partial**: `service.stop()` in lifespan; notifier stop; clip flush on shutdown not verified | Test SIGTERM: in-flight clips and notifications flushed |
| Prometheus conventions | Base units, histograms, `_total` | **partial**: `edgecopilot_frame_latency_ms{quantile}` gauge, reconnects as Gauge | `edgecopilot_frame_latency_seconds` histogram; reconnects as counter `_total` |
| OpenTelemetry | Traces for alert path | **missing** | Optional `opentelemetry-instrument` + span around frame→alert→store→notify (P2) |
| SRE SLOs | SLIs/SLOs, burn-rate alerts | **missing**: threshold alerts in `deploy/prometheus/alerts.yml` but no SLO definitions | `docs/SLO.md`: camera availability ≥ 99.5 %, alert delivery ≤ 30 s for 99 %, notification success ≥ 99 %; burn-rate rules; synthetic test alert hourly per camera |
| SRE runbooks | Per-alert runbook, incident review | **met/partial**: `docs/OPERATIONS.md` failure-mode table and incident steps | Link each Prometheus alert to a runbook anchor; post-incident review template |
| Model cards (Mitchell) | Factors, disaggregated metrics, out-of-scope use | **partial**: `models/README.md` has source, licence, I/O, intended use, known limits; no disaggregation (day/night/IR, camera height, Indian campus scenes), no CIs | Add factors + disaggregated results + CIs; out-of-scope uses (not a fire alarm, not identification); system card for the whole copilot |
| SBOM (CycloneDX/SPDX) / ML-BOM | SBOM per release incl. models | **missing** | CI: `cyclonedx-py` (or syft on the image) + ML-BOM entries for ONNX files with sha256, licence, pedigree |
| SLSA | Provenance (L1) → signed provenance (L2) | **missing** (L0): CI builds the image but publishes no provenance | GitHub `actions/attest-build-provenance` on image and wheel; pin actions by SHA |
| Dependency hygiene | Locked, hashed deps; vuln scan | **partial**: `pip-audit`, bandit, ruff, mypy in CI; version ranges only, no lockfile; base image tag pinned but not digest | `uv lock` / `pip-compile --generate-hashes`; pin `python:3.11.11-slim-bookworm@sha256:…`; Dependabot |
| DPDP Rule 6(a) | Encryption / masking | **met**: Fernet-encrypted media; privacy filter (blur/skeleton) before storage; keypoints not logged | Consider SQLCipher or filesystem encryption for the DB (alert metadata is indirect personal data) |
| DPDP Rule 6(b)/(c) | Access control + access logs | **met**: RBAC; clip/report access audited | Also audit `GET /api/alerts/{id}` snapshot views |
| DPDP Rule 6(d) | Backups / continuity | **partial**: see backups row | Backup script + restore test |
| DPDP Rule 6(e) | Logs retained ≥ 1 year | **partial**: audit log has no deletion path (good), but no stated retention or off-box copy | Document 1-year log retention; ship audit export off-box |
| DPDP Rule 7 / CERT-In | Breach response (without delay + 72 h report to DPB; CERT-In 6 h) | **partial**: PRIVACY.md references the institution's procedure | Breach runbook with timers, contacts, templates (org approves) |
| DPDP rights (access/erasure, 90-day grievance) | Locate + erase an individual's data | **partial**: locate by time/camera; erase only by manual file deletion | Admin `DELETE /api/alerts/{id}/media` (audited, erasure reason) + export-for-access |
| DPDP notice / Sec. 7 basis / children | Signage, notice, lawful basis, Fourth Schedule for minors | **partial**: PRIVACY.md guidance; decision left to DPO | DPO decision, signage, grievance officer (org) |
| EU AI Act Art. 5 / Annex III 1 | No RBI, no emotion inference | **met**: no face ID or re-ID; tracks local per camera | Rename "crowd panic" to "abnormal crowd movement" in UI/text; never add face recognition |
| EU AI Act Annex III 5(d) | Avoid automated dispatch/prioritisation | **met (by design)**: recommendations only, human confirms | Keep it explicit: no auto-calls to 108/101; document in intended-use statement |

---

## 4. Prioritised engineering backlog for a campus pilot

### P0: do before any pilot (each a few hours or less; students can do these)
1. **Fire disclaimer everywhere** (README, console footer, fire alert text, notification text, incident report). 30 min.
2. **Faults reach a human**: route `camera_fault`, offline and pipeline exceptions through `Notifier` escalation. Add Alertmanager to compose wired to the same webhook/email. Add a console "degraded / not monitoring" banner. 2–3 h.
3. **Software watchdog**: per-camera heartbeat, systemd `WatchdogSec` + `sd_notify`, `/readyz` reports last-frame age per camera. Export `edgecopilot_last_frame_age_seconds`. 1–2 h.
4. **Stop/pause control** (NIST MANAGE 2.4, AI Act Art. 14): admin-only pause/resume per camera + global maintenance mode, audited, with banner and metric. 2 h.
5. **Erasure + legal hold**: admin delete-media endpoint (audited) and a `protected` flag exempt from purge. 2 h.
6. **Time discipline**: UTC ISO-8601 with offset in DB, reports and logs; chrony to NIC/NPL in OPERATIONS.md; clock-skew check at start. 1 h.
7. **TLS in the shipped stack**: Caddy with campus CA in compose; MQTT 8883 + per-device credentials; ESP32 sketch sends `X-Api-Key` over HTTPS/MQTTS; self-host fonts. 2–3 h.
8. **FOV-change fault**: promote `scene_change` to a `fov_changed` camera fault; compare against a commissioning reference frame. 1 h.

### P1: makes the "industry-standard" claim credible (hours each)
9. **Supply chain**: lockfile with hashes, base image pinned by digest, CycloneDX SBOM + ML-BOM in CI, GitHub build-provenance attestation (SLSA L2-style), actions pinned by SHA. 2–3 h.
10. **Model integrity**: `models/manifest.json` (sha256, version, metrics, licence); refuse to load on mismatch; record model version in every alert (already partly done). Optional OpenSSF model-signing. 2 h.
11. **Backups**: `scripts/backup.sh` (SQLite online backup, key escrow note) + restore drill documented. 1–2 h.
12. **Drift and quality monitoring**: per-camera dismissal rate, alerts/hour, VERIFY ratio and calibration sample count as metrics, with alert rules (e.g. dismissal rate > 50 % over 24 h means recalibrate). 1–2 h.
13. **SLOs + burn-rate alerts**: `docs/SLO.md`, latency histogram in seconds, hourly synthetic test alert per camera through the whole notify chain. 2–3 h.
14. **Model / system card upgrade**: factors (day/night/IR, camera height and angle, crowd density, Indian campus scenes), disaggregated results with CIs, out-of-scope uses, licence blockers. 2 h.
15. **Risk register / FMEA** (ISO 23894 / NIST MAP) and **intended / prohibited use statement**. 2–3 h.
16. **Audible alarm and acknowledgement timing** in the console; record time to acknowledge as an SLI. 1–2 h.
17. **Wording**: "abnormal crowd movement" instead of "panic"; no emotion terms. 30 min.

### P2: nice to have
18. ONVIF discovery/GetStreamUri for commissioning; documented MQTT JSON alert schema (Profile-M-style) for VMS integration. Half a day.
19. OpenTelemetry tracing of the alert path. 2–3 h.
20. Release gate in CI: re-run the XAI and unit benchmarks on small fixtures; fail if metrics drop. 2–3 h.
21. DB-at-rest encryption (SQLCipher or LUKS on the edge box). 1–2 h plus ops.
22. 24 h soak test and multi-camera load test on the target Pi 5 / Jetson. 1 day, needs hardware.

### Needs a real organisation (cannot be closed in code)
- **Campus field validation**: weeks of footage from the real cameras with a ground-truth incident log. Measure false alarms per camera-day and misses. Includes a nuisance test set and supervised smoke tests with the fire officer.
- **DPDP**: DPO sign-off of the DPIA, choice of lawful basis (Sec. 7 vs notice/consent), signage and notices, grievance officer, retention policy, processor contracts, breach procedure (DPB without delay + 72 h; CERT-In 6 h), handling of minors under the Fourth Schedule.
- **ISO/IEC 42001 AIMS**: AI policy, roles, management review, internal audit, supplier assessment (Ultralytics weights, datasets). Certification is an organisational audit.
- **Security**: campus IT network segmentation, penetration test, CERT-In point of contact, key custody.
- **Operations**: named operator accounts, SOP integration with the security desk, operator training on automation bias and VERIFY vs ALARM, drills, monthly review of dismissals.
- **Life safety**: CFO/AHJ consultation. Any connection to the fire panel only through the campus fire contractor, and only as auxiliary/supervisory. Listing/approval (UL 268B / FM 3232 / VdS / future EN 54-34) is a manufacturer-level programme, out of scope.
- **Licences**: AGPL-3.0 weights and CC BY-NC training data are acceptable for a non-commercial academic pilot but block commercial deployment. Retrain on permissively licensed data, or license commercially.

---

## 5. Sources

Fire / life safety
- IS 2189:2008 text: https://law.resource.org/pub/in/bis/S03/is.2189.2008.pdf
- BIS record (IS 11360 referring IS 2189): https://services.bis.gov.in/php/BIS_2.0/bisconnect/knowyourstandards/Indian_standards/isdetails/MjQ0OA==
- NBC 2016 Part 4 summary: https://infralens.in/knowledge/nbc-2016-part-4-fire-safety ; Part 4 excerpt: https://nmcnagpur.gov.in/assets/300/2025/03/mediafiles/NBC2016-Part-IV.pdf
- NYC Buildings Bulletin 2014-005 (VISD): https://www.nyc.gov/assets/buildings/bldgs_bulletins/bb_2014-005.pdf
- NFPA 72 VISD/VIFD provisions (Campus Safety): https://campussafetymagazine.com/news/fire-side-chat-video-can-prove-valuable-in-fire-applications/16847/2
- FM Approvals Class 3232 (2023): https://www.fmapprovals.com/-/media/Feature/Approval-Standards/3232-pdf.pdf ; https://www.fmapprovals.com/products-we-certify/products-we-certify/detection-fire-alarm-and-signaling-equipment/video-image-fire-and-smoke-detectors
- CAN/ULC-S592 VISD notice: https://scc-ccn.ca/standards/notices-of-intent/ulc-standards/standard-video-image-smoke-detection-system-visd-devices ; https://scc-ccn.ca/standards/notices-of-intent/ulc-standards/video-image-smoke-detection-system-devices
- ISO/TS 7240-29: https://www.iso.org/news/ref2207.html ; FIA fact file on ISO/TS 7240-30: https://fia.uk.com/static/77040b07-26a7-440d-b81f1041494c2b47/Fact-File-98-Video-Fire-Detection-Systems.pdf
- prEN 54-34 (video fire detectors, draft): https://www.din.de/en/wdc-proj:din21:380549258
- Security Systems News on video fire detection (2006, historical): https://www.securitysystemsnews.com/article/next-wave-fire-video-smoke-and-flame-detection

Video surveillance / cameras
- IEC 62676-1-1 conformance summary (Bosch BVMS): https://community.boschsecurity.com:443/varuj77995/attachments/varuj77995/bt_community-tkb-video/355/1/BVMS%20-%20IEC62676-1.pdf
- IS 16910 = IEC 62676 adoption: https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/knowyourstandards/Indian_standards/isdetails_mnd/23091
- ONVIF Profile M: https://www.onvif.org/wp-content/uploads/2022/05/onvif-profile-m.pdf
- India CCTV ER / STQC: https://www.matrixcomsec.com/stqc-certification-er-compliance-for-cctv-cameras/

AI governance
- ISO/IEC 42001 Annex A: https://riskprofs.com/iso-42001-annex-a-controls-list/
- ISO/IEC 23894: https://blog.ansi.org/ansi/iso-iec-23894-2023-ai-risk-management/
- NIST AI RMF MANAGE: https://docs.modulos.ai/frameworks/nist-ai-rmf/manage
- EU AI Act Annex III: https://artificialintelligenceact.eu/annex/3/ ; biometrics guidance: https://ai-act-service-desk.ec.europa.eu/de/node/461
- Art. 12: https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-12 ; Art. 14: https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-14
- Emotion-recognition ban (FPF): https://fpf.org/blog/red-lines-under-eu-ai-act-unpacking-the-prohibition-of-emotion-recognition-in-the-workplace-and-education-institutions/
- AI Omnibus: https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/ ; https://natlawreview.com/article/eu-digital-omnibus-ai-enters-force
- India AI Governance Guidelines: https://babl.ai/india-releases-landmark-ai-governance-guidelines-emphasizing-safety-accountability-and-inclusion/

Privacy / India
- DPDP Rules 2025 (EY alert): https://www.ey.com/content/dam/ey-unified-site/ey-com/en-in/alerts-hub/2025/11/digital-data-protection-act-rules-notified-by-meity.pdf
- Rule 6: https://dpdpa.com/dpdparules/rule6.html ; Rule 7: https://dpdpa.com/dpdparules/rule7.html ; Rule 12: https://dpdpa.com/dpdparules/rule12.html
- Section 7 legitimate uses: https://dpdp.myndsolution.com/wiki/act/section-7-certain-legitimate-uses/
- Fourth Schedule: https://dpdp.myndsolution.com/wiki/rules/schedule-4-classes-of-data-fiduciaries-in-respect-of-whom-provisions-of/
- CERT-In Directions 2022: https://ankura.com/insights/everything-you-need-to-know-about-indias-new-guidelines-related-to-cyber-incident-reporting-by-cert-in/

Security / quality
- OWASP API Top 10 2023 (ISACA): https://www.isaca.org/resources/news-and-trends/industry-news/2023/reviewing-the-2023-owasp-api-top-10
- OWASP ASVS 5.0: https://softwaremill.com/whats-new-in-asvs-5-0/
- IEC 62443 foundational requirements: https://winccoa.com/documentation/WinCCOA/3.18/en_US/iec_62443/topics/iec_62443_foundational_requirements.html
- ISO/IEC 25010:2023: https://quality.arc42.org/standards/iso-25010 ; https://quality.arc42.org/articles/iso-25010-update-2023

MLOps / supply chain / ops
- Google MLOps levels: https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- Model cards: https://ar5iv.arxiv.org/html/1810.03993
- CycloneDX ML-BOM guide: https://cyclonedx.org/guides/OWASP_CycloneDX-Authoritative-Guide-to-AI-ML-BOM-en.pdf ; https://wavect.io/blog/ai-bill-of-materials-cyclonedx-spdx-2026/
- SLSA levels: https://slsa.dev/spec/v1.0/levels
- OpenSSF model signing: https://blog.sigstore.dev/model-transparency-v1.0
- SRE alerting on SLOs: https://sre.google/workbook/alerting-on-slos/
- Prometheus naming: https://prometheus.io/docs/practices/naming/
- OpenTelemetry Python: https://opentelemetry.io/docs/languages/python/ ; https://opentelemetry.io/docs/zero-code/python/
- Twelve-factor: https://12factor.net/
