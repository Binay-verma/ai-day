# Build log

## Context
- Built overnight on a 2-core CPU, 7.8 GB RAM, no GPU; only GitHub + PyPI reachable.
- GitHub push was not possible from the build box (no account linked): the repo is
  delivered as a git bundle + zip; push with the commands in the hand-over note.
- Research: docs/research/{landscape,methods}.md

## Done
- [x] Research (≈95 sources) and PLAN.md with acceptance criteria
- [x] Core XAI: evidence ledger, counterfactuals, grounded NLG + verifier, triage, conformal gate
- [x] Perception: ONNX YOLO runtime, ByteTrack-style tracker, fall / crowd / smoke cues, ego-motion
- [x] Pipeline, FastAPI server + operator console, CLI, Prometheus metrics, Docker, CI, 23 tests
- [x] Fire/smoke detector fine-tuned on CPU (mAP50 0.38 → 0.73), FP32 + INT8 ONNX
- [x] Benchmarks: falls (GMDCSA24), crowd (UMN), fire images + FURG video, XAI, saliency, latency
- [x] docs/RESULTS.md with protocols, numbers, failure analysis

## Open items (honest gaps)
- Fall F1 0.83 cross-subject vs 0.90 target: misses falls onto beds / partly out-of-frame bodies;
  lying down on a bed is the main false alarm (routed as VERIFY). Next: a stronger pose model
  (e.g. 640-px YOLO26s-pose) or a small temporal model over key-points.
- UMN scene 2 (indoor) without per-camera calibration; fire: tiny flames in FURG house3/4/6.
- No on-device LLM narrator could be downloaded; the verifier was stress-tested with
  synthetic narrator errors instead.
