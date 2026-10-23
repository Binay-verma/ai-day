# AI governance for the campus pilot (ISO/IEC 42001 & NIST AI RMF starter)

## Intended use and limits

* **Intended use**: decision support for trained campus security staff, to notice
  falls, smoke/fire and abnormal crowd movement (sudden dispersal, rapid exit,
  over-capacity) sooner, and to follow the campus emergency playbook.
* **Not intended for**: replacing the certified fire alarm system (IS 2189 /
  NBC 2016), medical diagnosis, identifying or tracking individuals, emotion
  recognition, disciplinary use, attendance, or automatically dispatching
  emergency services. "Abnormal crowd movement" describes motion patterns, not
  emotions.
* **Human oversight**: every alert is a recommendation for a human. Operators
  can confirm/dismiss (recorded), ask "why", and pause analysis for a bounded
  time (always audited, auto-resumes).

## Roles (RACI)

| Activity | Project lead | Model owner | Safety officer | Security desk operators | IT security | DPO |
|---|---|---|---|---|---|---|
| Approve deployment & intended use | A | C | R | I | C | C |
| Model updates (retrain, thresholds, weights) | I | R/A | C | I | C | I |
| Camera commissioning & calibration | I | C | A | R | I | I |
| Responding to alerts | I | I | A | R | I | I |
| Access control, keys, patching | I | I | I | I | R/A | I |
| Privacy notices, DPIA, data-subject requests | I | I | C | I | C | R/A |
| Quarterly performance & bias review | A | R | R | C | I | C |

## Change management

1. Any change to `configs/*.yaml` hazard weights, thresholds or playbooks, or to model files, goes through a pull request.
2. CI must pass (tests, lint, type, security scans, config lint).
3. Re-run the benchmarks in `benchmarks/` and compare with `docs/RESULTS.md`; regressions must be justified in the PR.
4. Pin the new model hashes (`scripts/make_sbom.py`) and record the change in `CHANGELOG.md`.
5. Re-calibrate affected cameras; announce the change to operators.

## Monitoring obligations

* Weekly: operator precision per hazard and camera, dismissed-alert reasons.
* Monthly: audit-chain verification, retention purge, missed-incident reports (incidents found by other means).
* Quarterly: risk register review (docs/RISK_REGISTER.md), fairness check (are alerts concentrated on particular areas or groups?), drill with the fire officer.

## Reporting concerns

Anyone (students, staff) can report a concern about the system to the safety
officer or the DPO; reports are logged and answered within 15 working days.
