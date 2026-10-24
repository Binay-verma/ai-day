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
| Not evaluated (factors) | night / IR, top-down ceiling cameras, wheelchairs and mobility aids, children, very crowded scenes (> 50 people) |
| Out of scope | face recognition, re-identification across cameras, emotion inference: tracks are local to one camera and are never linked to identity |

## `yolo11s-pose.onnx` — larger pose model for fall-critical cameras

| | |
|---|---|
| Source | Ultralytics YOLO11s-pose (COCO-pose), 9.9 M parameters, AGPL-3.0 |
| Used by | `configs/profiles/fall_accurate.yaml`, or per camera via `cameras[].pose` |
| Evidence | same cues and expert weights as the default: UR-Fall (never used in development) 23 / 30 falls vs 19 / 30 with YOLO11n-pose; external clips 9 / 11 vs 6 / 11, no false fall alarm on 7 minutes of walking footage |
| Cost | ~3x the CPU of YOLO11n-pose (4.2 FPS for all three hazards on 2 cores at 640 px) |
| Do not | combine it with fall weights re-fitted on one dataset: that learned a single-cue shortcut (17 false alarms, RESULTS section 8); `min_cues: 2` in the config now rejects such weights |

## `fire_smoke.onnx` — fire and smoke detector

| | |
|---|---|
| Start point | YOLO11n trained on D-Fire by sayedgamal99 (`Real-Time-Smoke-Fire-Detection-YOLO11`, MIT); class order 0 = smoke, 1 = fire |
| Fine-tuning | `scripts/train_fire.py` on the **leakage-free v2 split** (`scripts/prepare_fire_data.py`): Home-fire train (indoor, CC BY-NC 4.0) + DBA-Fire scenes (pHash scene grouping, whole scenes split 80/20) + 600 DeepQuest "Neutral" hard negatives, minus every image within 10 pHash bits of any evaluation image (2,211 removed); 416 px, backbone frozen (10 layers), CPU |
| Held-out evaluation | Home-fire test (1,300), DBA-Fire scene hold-out (665 images, 496 scenes), DeepQuest Neutral negatives (400, never trained on), FURG fire videos (system level). Manifest: `data/fire_splits.json` |
| Outputs | `output0` (1, 6, N) boxes + 2 class scores; P3 neck feature for Eigen-CAM |
| Intended use | Early visual cue for smoke/fire inside a multi-cue evidence ledger together with dark-channel smoke density, flame flicker and MQ-2/temperature sensors — not a stand-alone fire alarm |
| Known limits | Sun glare, red/orange lighting and steam can look like flame/smoke (neutral-image false-positive rate is above the 10 % plan budget, see RESULTS §4); very small distant smoke is missed |
| Not evaluated (factors) | night / IR cameras, ceiling-mounted fisheye, Indian campus kitchens and labs (steam, welding, sodium lamps), long-range outdoor smoke. Validate on campus footage before relying on it |
| Out of scope | a stand-alone or certified fire alarm (IS 2189 / NFPA 72 / EN 54), automatic dispatch, any use on people (identification, attendance, discipline) |
| Licence blockers | Home-fire training data is CC BY-NC 4.0 and the Ultralytics architecture/weights are AGPL-3.0: fine for a non-commercial academic pilot; commercial use needs retraining on permissive data or a commercial licence |

## INT8 variants (`*.int8.onnx`)

Static QDQ quantisation with per-channel weights, calibrated on 300 real frames,
detection-head decode kept in FP32 (`scripts/quantize.py`). Accuracy and speed
versus FP32 are reported in `docs/RESULTS.md`.

## Integrity

Every ONNX file can be pinned by SHA-256 in the config (`models.<name>.sha256`); the
runtime refuses to load a file whose hash differs. `scripts/make_sbom.py` lists the
models with hash, licence and pedigree in the CycloneDX ML-BOM.
