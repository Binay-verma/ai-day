# What we took from Argus, and what we did not

Argus (`KumarSrinidhi/Argus`, the team's earlier work on hackathon problem 5, "Intelligent Threat
Detection and Situational Awareness") is a broad security console: 24 k lines, RTSP/ONVIF cameras,
zone and flow rules, Re-ID across cameras, a VLM explainer, incident lifecycle. This copilot is
narrower and deeper: emergency response (falls, fire/smoke, crowd, intrusion) with exact,
verifiable explanations. Argus was compared with this project feature by feature; the ideas below
were **re-implemented here** to fit this architecture (no code was copied; Argus has no licence
file, so all rights stay with its authors).

## Taken (re-implemented, tested)

| Idea from Argus | Here | Why it matters for emergency response |
|---|---|---|
| Phone push via Telegram and ntfy, tokens scrubbed from errors | `notify.py`: `ntfy` and `telegram` channels in the existing escalation chain, `POST /api/notify/test` | The first-aid team or warden gets the explained alert on their phone; escalation stops when someone acknowledges |
| Evidence bundles: SHA-256 manifest, HMAC signature, server-side verify | `evidence.py`, `GET /api/alerts/{id}/evidence.zip`, `POST /api/evidence/verify`, `scripts/verify_evidence.py` | An incident can be handed to an insurer, the police or an inquiry and checked later; the verifier names exactly what changed |
| Operator track record per rule and camera with a Wilson interval | `track_record.py`, shown on every alert, `GET /api/track_record` | Trust in an alert comes from this site's own record, not from a claim |
| Natural-language investigation through an allow-listed query, never SQL, citing alert ids | `copilot/search.py`, `POST /api/search`, search box in the console | "falls in Corridor B last night" answered in a second, every number traceable to listed alerts |
| Privacy masks painted before detection, recording, snapshots and live view | `cameras[].masks`, `Copilot.apply_masks` | A neighbour's window or a washroom door is never recorded or analysed |

## Already here (Argus has an equivalent; no change needed)

Hash-chained audit log with verify, legal hold and retention, camera tamper / trouble alerts,
shift handover report, incident correlation across cameras, escalation of unacknowledged alerts,
role-based access, rate and body limits, CSP, per-device sensor keys bound to zones with
server-side timestamps (Argus signs door-panel events with HMAC; here the device key is stored only
as a digest and readings are stamped with the server clock, which already blocks stale or future
readings, so signing was not added).

## Deliberately not taken

| Argus feature | Why not |
|---|---|
| Re-ID across cameras, next-camera prediction, trajectory forecasting | Following a person across cameras is surveillance, and Argus's own results are research-grade (Re-ID precision 0.61 / recall 0.36). Instead, the emergency-relevant part was built without identity (RESULTS section 16): exit safety from the cameras that watch each exit, a measured heading predictor (where people are moving), "crowd heading into danger", and "went down and left the view". A per-exit flow count / time-to-clear estimate was built too but **discarded**: it could not be validated on the available footage |
| Weapon and altercation detectors | Out of scope; Argus measured 8 % weapon recall on CCTV views and AUC 0.68 for altercation |
| VLM explainer (Qwen2.5-VL) | Needs a GPU; here every sentence is generated from the exact ledger and checked by the verifier, which a VLM cannot guarantee |
| ONVIF discovery, vendor RTSP templates, WebRTC restream, remote GPU nodes | Deployment conveniences for large sites; useful later, not decisive now |
| Hour-of-week baselines | The crowd model already learns per-camera normal speed and calibrates its alert line from normal footage |

## Where this project is stronger than Argus

* **Falls:** Argus detects 0 of 3 CAVIAR falls (its fall feature is experimental). Here: GMDCSA24
  F1 0.883, UR-Fall 23-24 / 30 with 0 / 5 alerts on daily activities, and a fall evidence gate that
  separates lying down from falling with an exact, named reason (RESULTS section 2.1).
* **Fire and smoke:** Argus has no fire detection. Here: a leakage-free fire/smoke detector plus
  MQ-2 sensor fusion, 15 / 17 FURG fire videos with 0 false alarms.
* **Explanations:** exact additive attributions (error ~1e-15), counterfactuals, a What-if
  explorer and a grounding verifier that catches 100 % of 576 injected narrator errors.
