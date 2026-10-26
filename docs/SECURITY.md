# Security

## Threat model (STRIDE, campus pilot)

| Asset | Threat | Control in this repo |
|---|---|---|
| Operator console / API | **Spoofing**: someone on the campus LAN uses the console | Login required for every route except `/healthz` and static files; scrypt password hashes; HS256 JWT sessions in HttpOnly + SameSite=Strict cookies; refusal to start without auth on a non-loopback bind; brute-force lockout (5 failures / 5 min per user+IP) |
| Alert decisions | **Tampering**: a viewer dismisses alerts, or someone edits history | Roles viewer < operator < admin enforced per route; every confirm/dismiss/login/export/notification written to a **hash-chained audit log** (`GET /api/audit/verify` detects any edited or deleted row) |
| Sensor channel | **Spoofing/tampering**: fake MQ-2 readings suppress or trigger fire alerts | Per-device API keys (only SHA-256 digests stored); pydantic validation with physical ranges; finite numbers only; server-side time-stamps (devices cannot inject stale/future readings); MQTT payload validator with size limit; stale sensors drop to reliability 0; camera-vs-sensor disagreement routes to VERIFY rather than being trusted blindly |
| Video / snapshots | **Information disclosure** (personal data) | Privacy filtering before anything is stored or streamed (head blur from pose key-points or skeleton-only); media encrypted at rest (Fernet); retention purge; server paths never exposed by the API; media served only from the media directory (path-traversal guard) |
| Console (browser) | **XSS / clickjacking** | All server data rendered via `textContent`/DOM APIs (no `innerHTML`); CSP `script-src 'self'` (no inline script), `frame-ancestors 'none'`, `X-Frame-Options: DENY`, `nosniff`; incident reports HTML-escape every field |
| Service availability | **DoS** | 64 KB request-body limit; token-bucket rate limiting per client; bounded WebSocket clients and per-client queues (slow clients lose frames, never alerts); video frames throttled per client |
| Browser camera (`/ws/camera`, "Use my camera") | **Spoofing / DoS / disclosure**: an outsider feeds video, floods the server, or watches someone else's camera | Operator role and same-origin check; reasons sent after accept, then closed; per-IP connect rate limit; bounded sessions (`max_sessions`) and duration (`max_minutes`), 30 s idle timeout; frames must be JPEG, at most 1.5 MB and 64-3840 px, at most `max_fps`, one in flight; capture time cannot run ahead of the server clock; the live view goes back to the sender only and is not listed or exposed as a camera; faces are blurred as for any camera; sessions are audited (user, IP, frames, duration) |
| Playbooks / config | **Elevation of privilege** via config injection | Playbook conditions evaluated by a whitelisting AST evaluator (no calls, attributes, comprehensions); config schema validation refuses unknown keys and unknown names at start-up; `yaml.safe_load` only |
| Notifications | Leakage over plain HTTP | Webhooks must be `https://` (localhost excepted); SMTP with STARTTLS; credentials only from environment variables |
| Supply chain | Vulnerable dependencies | `pip-audit` in CI, `bandit` static analysis, pinned base image, non-root container with read-only filesystem, dropped capabilities |

## Secrets

| Secret | Where | Rotation |
|---|---|---|
| `EDGECOPILOT_SECRET` (JWT signing, ≥ 32 chars) | env / `.env` (else generated once into `data/.secret`, mode 0600) | rotate = all sessions end |
| `EDGECOPILOT_MEDIA_KEY` (Fernet) | env (else `data/.media_key`, 0600) | re-encrypt media or let retention expire it |
| `EDGECOPILOT_METRICS_TOKEN` | env; Prometheus `credentials_file` | any time |
| Device keys | shown once by `python -m edgecopilot.security keygen`; only digests in config | per device |
| SMTP / MQTT credentials | env vars named in config | per provider policy |

## Hardening checklist for a pilot

- [ ] Put the console behind HTTPS (reverse proxy such as Caddy/nginx with a campus certificate); the session cookie becomes `Secure` automatically behind `X-Forwarded-Proto: https`.
- [ ] Create named accounts for every operator (`python -m edgecopilot.security hash`), no shared logins.
- [ ] Place cameras, the edge box and the broker on an isolated VLAN; block outbound internet except notification endpoints.
- [ ] Enable MQTT TLS (port 8883) and per-device broker passwords.
- [ ] Back up `data/edgecopilot.db` daily (it contains no images); keep keys in a separate secret store.
- [ ] Review `GET /api/audit/verify` weekly; export the audit log for the safety committee.

## Reporting a vulnerability

Email the maintainers (team local maxima) with details; do not open a public issue.
