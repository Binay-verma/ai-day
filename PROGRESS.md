# Overnight build log (resume point)

If a session resumes, read this file first, then continue the first unchecked item.

## Context
- Workspace: 2-core CPU, 7.8 GB RAM, no GPU. Network: only GitHub + PyPI reachable.
- GitHub push not possible (no account linked) -> commit locally, ship bundle/zip.
- Research outputs: /home/claude/research/{landscape,datasets_models,methods}.md
- Datasets downloaded by research agent: /home/claude/datasets/*

## Status
- [x] Repo skeleton, core XAI (fusion, counterfactual, NLG + verifier, calibration)
- [x] Perception: detector (ONNX), tracker, fall cues, crowd flow cues, smoke cues
- [x] Pipeline, sensors hub, recommender (safe playbook conditions), events, privacy, Q&A
- [ ] Research reports in, PLAN.md written
- [ ] Models: yolo11n-pose ONNX (+feature output for Eigen-CAM), fire/smoke ONNX
- [ ] Server + dashboard
- [ ] Benchmarks: fall, fire/smoke, crowd, latency, XAI faithfulness, NLG grounding, conformal FAR
- [ ] Fix loop until acceptance criteria pass
- [ ] Tests + CI, README, results report, final commit + bundle
