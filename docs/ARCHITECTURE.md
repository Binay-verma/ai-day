# Architecture

## Runtime view

```
                 ┌───────────────────────────── edge box (2-8 CPU cores) ─────────────────────────────┐
 RTSP/IP camera ─┤ CameraWorker (thread per camera)                                                    │
 RTSP/IP camera ─┤   reconnect + back-off · frame dropping under load · tamper checks · clip ring buffer│
                 │        │                                                                            │
                 │        ▼                                                                            │
                 │   Copilot (one per camera / zone)                                                   │
                 │     motion gate ─► YOLO11n-pose (ONNX) ─► ByteTrack-style tracker ─► fall cues      │
                 │                └► fire/smoke YOLO11n (ONNX) + dark-channel/flicker ─► fire cues     │
                 │     Farneback flow ─► surge / exodus / divergence / coherence ─► crowd cues        │
                 │     SensorHub (shared) ◄── MQTT bridge ◄── ESP32 nodes (MQ-2, DHT22, sound)          │
                 │                       ◄── POST /api/sensors (device keys)                          │
                 │     evidence ledger (additive log-odds) ─► hysteresis ─► conformal alert line       │
                 │     ─► triage (ALARM/VERIFY) ─► counterfactuals ─► template text ─► verifier         │
                 │     ─► SOP recommender ─► Alert                                                    │
                 │        │                                                                            │
                 │        ▼                                                                            │
                 │   EdgeService: Store (SQLite WAL, encrypted media, hash-chained audit, retention)    │
                 │                Notifier (webhook/Slack/Teams/e-mail/MQTT, retries, escalation)      │
                 │                listeners ─► FastAPI server (RBAC, WebSocket fan-out, reports)        │
                 │                metrics ─► /metrics (Prometheus)                                     │
                 └──────────────────────────────────────────────────────────────────────────────────────┘
                          │ HTTPS (reverse proxy)          │ scrape                 │ webhooks / SMTP
                    operator console                 Prometheus + Grafana       security desk, wardens
```

## Module map

| Package | Responsibility | Key design decisions |
|---|---|---|
| `edgecopilot/perception` | ONNX YOLO runtime (detect + pose + neck feature), tracker, cue extractors | No PyTorch at runtime; every cue has a physical unit; reliabilities from key-point confidence |
| `edgecopilot/xai` | ledger fusion, counterfactuals, NLG + verifier, triage, calibration, saliency | The explanation *is* the model (additive); closed-form counterfactuals; verifier gates any text |
| `edgecopilot/copilot` | SOP recommender (safe expression evaluator), in-memory events, Q&A | Playbooks are data, not code |
| `edgecopilot/pipeline.py` | per-frame loop for one camera | stateless across cameras; scene-cut / exposure-jump / ego-motion handling |
| `edgecopilot/cameras.py` | stream lifecycle, health, tamper, clips | live sources always use the newest frame |
| `edgecopilot/service.py` | wiring: store, notifier, MQTT, cameras, provenance | storage/notification failures never block the video thread |
| `edgecopilot/storage.py` | SQLite, audit chain, encrypted media, retention | append-only audit with SHA-256 chain |
| `edgecopilot/security.py` | scrypt, JWT, roles, device keys, rate limiting | auth mandatory on non-loopback binds |
| `edgecopilot/server` | API, console, metrics | CSP without inline script; DOM-only rendering |
| `edgecopilot/config_schema.py` | typed config + cross-checks | the service refuses to start on any config error |

## Data flow of one alert

1. Frame → cues → ledger → smoothed score crosses the (calibrated) line on `on_frames` consecutive frames.
2. `Copilot._make_alert`: recommendations, triage, explanation, verification, counterfactuals, evidence map, privacy-filtered snapshot.
3. `EdgeService._on_alert`: encrypt + store media, save record with provenance (code version, config hash, model hashes), audit, notify (escalation chain), broadcast to consoles, mark the clip buffer.
4. Operator confirms/dismisses → store + audit → online bias update and conformal recalibration for that zone.

## Performance budget (2-core x86, FP32 640 px, all hazards)

| Stage | p50 ms |
|---|---|
| pose model | 59 (26 at 416 px, 46 INT8) |
| fire model (every 2nd frame) | ~55 when run |
| optical flow + crowd cues | 15 |
| fall cues, tracker, fusion, gate | < 2 |
| explanation + verification (on alert only) | 0.6 |
