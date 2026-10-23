# Overnight build log (resume point)

If a session resumes, read this file first, then continue the first unchecked item.

## Context
- Workspace: 2-core CPU, 7.8 GB RAM, no GPU. Network: only GitHub + PyPI reachable.
- GitHub push not possible (no account linked) -> commit locally, ship bundle/zip.
- Research outputs: docs/research/{landscape,methods}.md (+ /home/claude/research/datasets_models.md)
- Datasets: /home/claude/datasets/* ; caches: data/gmdcsa24_lowconf.pkl, data/umn_features.pkl
- Fire model: fine-tuned 4 epochs (runs/detect/runs/fire/ft/weights/best.pt -> models/fire_smoke.{pt,onnx,int8.onnx})

## Status
- [x] Core XAI (ledger, counterfactuals, NLG + verifier, triage, calibration/conformal gate)
- [x] Perception (ONNX YOLO, ByteTrack-style tracker, fall/crowd/smoke cues), pipeline, server, console UI, CLI
- [x] Research + PLAN.md
- [x] Fall benchmark (GMDCSA24): expert F1 0.814, LOSO bias-tuned 0.828, conformal FAR 0.099 <= alpha 0.1
- [x] Crowd benchmark (UMN): per-camera conformal 10/11 events, 0 false alarms; tolerant AUC 0.91
- [x] XAI benchmark: template grounding 100 %, verifier detection 99.2 %, FRR 1.2 %, CF validity 100 %
- [x] Fire image benchmark FP32 (mAP50 0.73 / 0.73 vs 0.36 / 0.30 baseline)
- [ ] Fire baseline + INT8 image eval, FURG video eval (logs/fire_eval.log, logs/fire_furg.log)
- [ ] Saliency faithfulness (Eigen-CAM deletion/insertion) benchmark
- [ ] Latency benchmark (benchmarks/bench_latency.py) on demo reel
- [ ] Demo run with fire model; dashboard screenshot check
- [ ] docs/RESULTS.md, README results table, final commit, git bundle + zip
