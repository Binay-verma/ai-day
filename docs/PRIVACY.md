# Privacy and data protection (DPIA summary)

*Engineering summary to support a Data Protection Impact Assessment by the
campus. It is not legal advice; the institution's data-protection officer
must review it against the Digital Personal Data Protection Act, 2023 and the
DPDP Rules in force at deployment time.*

## What personal data exists, where, and for how long

| Data | Personal? | Where it lives | Retention | Protection |
|---|---|---|---|---|
| Live video frames | Yes | RAM only, never written to disk | seconds | processed on the edge box, never leaves it |
| Live stream to the console | Yes (reduced) | WebSocket to logged-in users | not stored | privacy filter applied first: heads blurred using pose key-points (`blur`) or people reduced to skeletons on a blurred background (`skeleton`) |
| Alert snapshot + evidence image | Yes (reduced) | `data/media/*.enc` | `storage.retention_days` (default 30) | privacy-filtered, then encrypted (Fernet, AES-128-CBC + HMAC-SHA256) |
| Pre/post-event clip (≈15 s) | Yes (reduced) | `data/media/*_clip.mp4.enc` | same | privacy-filtered, encrypted, downloadable only by operators/admins, every download audited |
| Alert record (hazard, time, cue values, explanation) | Indirect (time + place) | SQLite `alerts` | metadata kept for safety statistics; subject id and free-text notes redacted at retention | access by role |
| Operator accounts and actions | Yes (staff) | config (hashes only) + audit log | per institution policy | scrypt hashes; hash-chained audit log |
| Sensor readings (smoke ppm, °C, dB) | No | RAM (rolling window) | minutes | — |

No face recognition, no identity, no biometric template, no tracking across
cameras, no audio recording (the sound sensor reports a level in dB only).
Track ids (`person#3`) are local to one camera and reset continuously.

## Purpose limitation and lawful basis (to be confirmed by the DPO)

The purpose is the safety of people on campus: detecting falls, fire/smoke and
dangerous crowding so that help arrives sooner. The DPDP Act allows processing
for certain *legitimate uses*, including responding to medical emergencies and
ensuring safety during disasters or breakdowns of public order (Section 7);
whether a given deployment relies on legitimate use or on notice and consent is
for the institution to decide and document. Either way: put up CCTV/AI notices
in monitored areas, name a contact for questions and grievances, and do not
reuse the data for discipline, attendance or other unrelated purposes.

## Data-subject rights and breach handling

* **Access / erasure requests**: incidents can be located by time and camera
  (`GET /api/alerts?camera=…`); media of an incident can be removed by
  shortening retention or deleting the encrypted files; the action is audited.
* **Breach**: media are encrypted with a key held outside the database; the
  audit log shows who accessed what. The institution's breach-notification
  procedure (Data Protection Board and affected persons) applies.

## Privacy-by-design choices

1. Edge processing: raw video never leaves the device.
2. Data minimisation: only privacy-filtered images are stored, and only for alerts.
3. Pose key-points are themselves treated as personal data (gait/skeleton can
   re-identify people), so they are not logged or exported.
4. Storage limitation: automatic hourly retention purge (audited).
5. Human oversight: alerts are recommendations; VERIFY routing for weak or
   conflicting evidence; operator decisions recorded with names.
