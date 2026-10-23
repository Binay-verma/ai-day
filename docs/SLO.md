# Service level objectives

| SLI | SLO | Measured by | Alert |
|---|---|---|---|
| Camera availability (stream live and analysed) | ≥ 99.5 % per camera per 30 days | `edgecopilot_camera_up` | `CameraNotLive` (2 min), `CameraAvailabilitySLOBurn` |
| Hazard-to-alert time (onset → alert on console) | ≤ 30 s for 99 % of confirmed incidents (FM 3232-style target) | benchmarks (fall median 1.0 s, crowd 0.8 s, fire 2.8 s on FURG) + incident reports | review monthly |
| Notification delivery | ≥ 99 % delivered within 60 s | `edgecopilot_notifications_total{ok}` | `NotificationFailures` |
| Processing latency | p95 ≤ 400 ms per frame | `histogram_quantile(0.95, rate(edgecopilot_frame_latency_seconds_bucket[5m]))` | `CopilotSlow` |
| Sensor freshness | every node reports at least every 10 s | `edgecopilot_sensor_age_seconds` | `SensorStale` |
| Operator precision (confirmed / (confirmed + dismissed)) | ≥ 0.5 per camera and hazard (alarm-fatigue guard) | `edgecopilot_operator_feedback_total{camera,hazard,outcome}`, `/api/stats?hours=168` | `HighDismissalRate` (24 h) |
| Frame freshness | every unpaused camera analysed within 30 s | `edgecopilot_last_frame_age_seconds`, `edgecopilot_camera_paused` | `CameraFramesStale`, `CameraPausedLong` |
| Calibration drift | alert line below its ceiling (`max_threshold`) | `edgecopilot_alert_threshold`, `edgecopilot_calibration_samples` | `AlertLineAtCeiling` |
| Audit integrity | `GET /api/audit/verify` = ok | admin check | weekly review |

Error budget policy: when a camera burns > 50 % of its monthly availability
budget, freeze configuration changes for that camera and fix the root cause
(power, network, mounting) first.
