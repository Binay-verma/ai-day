# Risk register (FMEA) — ISO/IEC 23894 / NIST AI RMF "Map & Manage"

Scores: Severity (S), Occurrence (O), Detection (D) on 1–5 (5 = worst / most
likely / hardest to detect); RPN = S×O×D. Re-score after the campus pilot.

| # | Failure mode | Effect | S | O | D | RPN | Controls in place | Further treatment | Owner |
|---|---|---|---|---|---|---|---|---|---|
| R1 | Missed fire / smoke (false negative) | delayed evacuation | 5 | 3 | 4 | 60 | certified fire alarm remains primary (disclaimers everywhere); multi-cue ledger incl. MQ-2 + temperature; FURG recall 13/17 measured | pilot with theatrical smoke; add thermal camera cue; keep sensors per room | safety officer |
| R2 | Missed fall | delayed first aid | 5 | 3 | 4 | 60 | pose cues + ByteTrack low-confidence continuation; motionless-time escalation to ambulance step | camera placement guide (full body visible); stronger pose model (evaluated) | model owner |
| R3 | False evacuation trigger | panic, injuries, alarm fatigue | 4 | 2 | 2 | 16 | recommendations only, never automatic dispatch; ALARM vs VERIFY triage; conformal false-alarm budget; operator confirm | drills; quarterly review of dismissed alerts | security desk lead |
| R4 | Alarm fatigue (many false alerts) | operators ignore real alerts | 4 | 3 | 2 | 24 | per-camera calibration, dismiss-feedback recalibration, re-alert cooldown, `AlertRateDrift` rule | weekly precision review via `/api/stats` | security desk lead |
| R5 | Camera down / covered / moved | blind spot | 5 | 3 | 1 | 15 | trouble-signal alerts (offline, frozen, blackout, defocus, view changed) through the same escalation chain; Prometheus `CameraNotLive` + Alertmanager | spare cameras; manual patrol procedure | facilities |
| R6 | Copilot process hangs / crashes | all zones blind | 5 | 2 | 1 | 10 | systemd watchdog (restart if a camera thread dies), Docker health check, `CopilotDown` alert | UPS for edge box and switch | IT |
| R7 | Sensor spoofing / stale sensor | wrong fire evidence | 4 | 2 | 2 | 16 | device keys, MQTT ACL + TLS, range validation, server time-stamps, staleness → reliability 0, camera/sensor disagreement → VERIFY | per-device certificates | IT security |
| R8 | Model file tampering / wrong model | silent accuracy loss | 4 | 1 | 2 | 8 | SHA-256 pinning (refuse to load), provenance hashes in every alert, SBOM/ML-BOM | signed releases (SLSA L2) | model owner |
| R9 | Unauthorised access to video/incidents | privacy breach | 4 | 2 | 2 | 16 | RBAC, scrypt, lockout, encrypted media, audit chain, retention purge, privacy filter before storage | HTTPS proxy, VLAN, pen test | IT security |
| R10 | Explanation misleads operator | wrong action | 3 | 2 | 2 | 12 | exact ledger, grounding verifier (0 % unsupported claims), counterfactuals, reliability shown | operator training with real examples | trainer |
| R11 | Distribution shift (new building, lighting, uniforms) | accuracy drops | 4 | 3 | 3 | 36 | per-camera calibration, ego-motion/exposure handling, drift alert | re-calibrate after changes; collect campus validation set | model owner |
| R12 | Misuse for surveillance of individuals | rights violation | 4 | 2 | 3 | 24 | no identity/face recognition, track ids local and short-lived, purpose limitation in PRIVACY.md, audit log | AI use policy (AI_GOVERNANCE.md), DPO review | DPO |
| R13 | Licence breach (AGPL weights, CC BY-NC data) in a commercial deployment | legal | 3 | 2 | 1 | 6 | licences documented in models/README.md and SBOM | retrain on permissively licensed data before any commercial use | project lead |
