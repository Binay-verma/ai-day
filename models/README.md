# Models (model cards)

Model binaries are not committed (size + AGPL weights); `python scripts/setup_models.py`
recreates them. The edge runtime only loads the ONNX files.

## `yolo11n-pose.onnx` — people + 17 COCO key-points

| | |
|---|---|
| Source | Ultralytics YOLO11n-pose (COCO-pose), `github.com/ultralytics/assets` release v8.4.0 |
| Licence | AGPL-3.0 (Ultralytics) |
| Input | 640×640 letterboxed RGB, FP32 |
| Outputs | `output0` (1, 56, 8400) boxes + score + key-points; `/model.16/cv2/act/Mul_output_0` (1, 64, 80, 80) P3 neck feature for Eigen-CAM |
| Used for | person tracking, fall cues (torso angle, descent, tilt rate, height drop, aspect, ground time), crowd head-count and scale, head anonymisation |
| Known limits | Misses people in unusual floor postures and when heavily occluded (see fall error analysis in `docs/RESULTS.md`); trained on COCO, not on CCTV angles |

## `fire_smoke.onnx` — fire and smoke detector

| | |
|---|---|
| Start point | YOLO11n trained on D-Fire by sayedgamal99 (`Real-Time-Smoke-Fire-Detection-YOLO11`, MIT); class order 0 = smoke, 1 = fire |
| Fine-tuning | `scripts/train_fire.py`: Home-fire train (indoor, CC BY-NC 4.0) + 80 % hash split of DBA-Fire + 600 DeepQuest "Neutral" hard negatives; 416 px, backbone frozen (10 layers), CPU |
| Held-out evaluation | Home-fire test, DBA-Fire 20 % split, DeepQuest Neutral test (never trained on), FURG fire videos (system level) |
| Outputs | `output0` (1, 6, N) boxes + 2 class scores; P3 neck feature for Eigen-CAM |
| Intended use | Early visual cue for smoke/fire inside a multi-cue evidence ledger together with dark-channel smoke density, flame flicker and MQ-2/temperature sensors — not a stand-alone fire alarm |
| Known limits | Sun glare, red/orange lighting and steam can look like flame/smoke; very small distant smoke is missed; licence of Home-fire restricts commercial use of the fine-tuned weights |

## INT8 variants (`*.int8.onnx`)

Static QDQ quantisation with per-channel weights, calibrated on 300 real frames,
detection-head decode kept in FP32 (`scripts/quantize.py`). Accuracy and speed
versus FP32 are reported in `docs/RESULTS.md`.
