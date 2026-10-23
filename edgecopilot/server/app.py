"""Operator API + console (FastAPI).

Security model (see docs/SECURITY.md):

* every route except ``/healthz`` and the static console requires a session
  (HttpOnly + SameSite=Strict cookie, or ``Authorization: Bearer``) with the
  right role: viewer < operator < admin; ``/api/sensors`` requires a device
  key; ``/metrics`` accepts a dedicated scrape token;
* strict security headers (CSP without inline script, no framing, no
  sniffing), request-size limit, per-client rate limiting, bounded WebSocket
  clients and queues, all inputs validated with pydantic;
* every privileged action is written to the hash-chained audit log.

Run:  python -m edgecopilot.server.app --config configs/campus.yaml
      python -m edgecopilot.server.app --source demo/demo_campus.mp4 --zone "Corridor B" --demo
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import html
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from .. import __version__
from ..config_schema import load_config_dict
from ..copilot.qa import answer
from ..logging_setup import setup as setup_logging
from ..security import AuthManager, Principal, RateLimiter, load_secret
from ..service import EdgeService, utc_iso
from . import metrics as prom

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
COOKIE = "ec_session"
MAX_BODY = 64 * 1024
log = logging.getLogger("edgecopilot.server")


# ------------------------------------------------------------------ schemas
class Login(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=1, max_length=256)


class Feedback(BaseModel):
    confirmed: bool
    note: str = Field("", max_length=500)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=300)


class Pause(BaseModel):
    minutes: float = Field(30, gt=0, le=12 * 60, description="bounded: analysis always resumes automatically")
    reason: str = Field("", max_length=200)


class WhatIf(BaseModel):
    values: Dict[str, float] = Field(default_factory=dict, max_length=32)


class SelfTest(BaseModel):
    notify: bool = False


class Sound(BaseModel):
    on: bool


class Hold(BaseModel):
    hold: bool


class SensorReading(BaseModel):
    zone: str = Field("default", min_length=1, max_length=60)
    smoke_ppm: Optional[float] = Field(None, ge=0, le=10000)
    temp_c: Optional[float] = Field(None, ge=-40, le=125)
    sound_db: Optional[float] = Field(None, ge=0, le=160)

    @field_validator("smoke_ppm", "temp_c", "sound_db")
    @classmethod
    def _finite(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("must be finite")
        return v


class BodyLimit:
    """Pure ASGI middleware: reject bodies over ``limit`` bytes while streaming.

    Checks Content-Length up front AND counts bytes of chunked/streamed bodies,
    so an anonymous client cannot make the server buffer hundreds of MB before
    authentication runs.
    """

    def __init__(self, app, limit: int = MAX_BODY):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        for k, v in scope.get("headers", []):
            if k == b"content-length" and v.isdigit() and int(v) > self.limit:
                return await self._reject(send)
        # Buffer at most `limit` bytes ourselves (64 KB), then replay the body to the app.
        # A chunked/streamed body without Content-Length is cut off as soon as it exceeds the limit.
        chunks, seen, more = [], 0, True
        while more:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            if msg["type"] != "http.request":
                continue
            body = msg.get("body", b"")
            seen += len(body)
            if seen > self.limit:
                return await self._reject(send)
            chunks.append(body)
            more = msg.get("more_body", False)
        data = b"".join(chunks)
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": data, "more_body": False}
            return await receive()  # afterwards: wait for disconnect like a normal server
        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(send):
        body = b'{"detail":"request too large"}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; font-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return resp


def client_ip(request) -> str:
    return request.client.host if request.client else "?"


def create_app(service: EdgeService, auth: AuthManager, cfg: dict) -> FastAPI:
    scfg = cfg.get("server", {})
    limiter = RateLimiter(scfg.get("rate_limit_per_min", 240))
    max_ws = scfg.get("max_ws_clients", 20)
    stream_period = 1.0 / scfg.get("stream_fps", 5.0)
    metrics_token = os.environ.get("EDGECOPILOT_METRICS_TOKEN", "")
    clients: set = set()
    loop_ref: dict = {}

    @asynccontextmanager
    async def lifespan(_app):
        loop_ref["loop"] = asyncio.get_running_loop()
        service.listeners.append(_fanout)
        service.start()
        prom.start_collector(service)
        yield
        service.stop()

    app = FastAPI(title="Explainable Edge AI Copilot", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SecurityHeaders)
    app.add_middleware(BodyLimit, limit=MAX_BODY)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        # never echo the offending input back (NaN/huge values made the error itself fail to serialise)
        return JSONResponse({"detail": [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "invalid")}
                                        for e in exc.errors()][:20]}, status_code=422)
    if scfg.get("cors_origins"):
        app.add_middleware(CORSMiddleware, allow_origins=scfg["cors_origins"], allow_credentials=True,
                           allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # --------------------------------------------------------- auth deps
    def _token(request) -> Optional[str]:
        h = request.headers.get("authorization", "")
        if h.lower().startswith("bearer "):
            return h[7:].strip()
        return request.cookies.get(COOKIE)

    def limit(request: Request, cost: float = 1.0, bucket: str = "api"):
        # separate buckets so a flood of sensor posts or logins cannot starve the operator console
        if not limiter.allow(f"{bucket}:{client_ip(request)}", cost):
            raise HTTPException(429, "too many requests")

    def need(role: str):
        def dep(request: Request) -> Principal:
            limit(request)
            p = auth.verify(_token(request))
            if p is None:
                raise HTTPException(401, "login required")
            if not p.can(role):
                raise HTTPException(403, f"requires role {role}")
            return p
        return dep

    viewer, operator, admin = need("viewer"), need("operator"), need("admin")

    # ------------------------------------------------------------ public
    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/healthz")
    def healthz():
        """Liveness: 503 if a camera thread died or a live camera stopped producing frames
        (container orchestrators / Docker HEALTHCHECK then restart the service)."""
        ok, why = service.liveness()
        return JSONResponse({"ok": ok, "version": __version__, **({"problem": why} if not ok else {})},
                            status_code=200 if ok else 503)

    @app.get("/api/openapi.json")
    def openapi_schema(p: Principal = Depends(viewer)):
        return app.openapi()

    @app.get("/readyz")
    def readyz():
        cams = [w.status() for w in service.cameras.values()]
        live = [c for c in cams if c["health"]["state"] == "live"]
        try:
            service.store.conn.execute("SELECT 1").fetchone()
            db = True
        except Exception:  # noqa: BLE001
            db = False
        ok = db and (not cams or bool(live))
        return JSONResponse({"ready": ok, "db": db, "cameras_live": len(live), "cameras": len(cams)},
                            status_code=200 if ok else 503)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request):
        tok = _token(request)
        if auth.enabled and not (metrics_token and tok and hmac.compare_digest(tok, metrics_token)) and auth.verify(tok) is None:
            raise HTTPException(401, "metrics token required")
        return PlainTextResponse(prom.render(), media_type="text/plain; version=0.0.4")

    # ------------------------------------------------------------ session
    @app.post("/api/login")
    def login(body: Login, request: Request, response: Response):
        limit(request, cost=5.0, bucket="login")
        ip = client_ip(request)
        tok = auth.login(body.username, body.password, ip)
        if tok is None:
            service.store.audit(body.username, "-", "login_failed", "", {"ip": ip})
            raise HTTPException(401, "invalid credentials or temporarily locked")
        p = auth.verify(tok)
        if p is None:  # cannot happen for a freshly issued token; keeps type-checkers honest
            raise HTTPException(500, "session error")
        service.store.audit(p.username, p.role, "login", "", {"ip": ip})
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(COOKIE, tok, httponly=True, samesite="strict", secure=secure, max_age=auth.ttl_s, path="/")
        return {"username": p.username, "role": p.role, "token": tok}

    @app.post("/api/logout")
    def logout(request: Request, response: Response, p: Principal = Depends(viewer)):
        tok = _token(request)
        if tok:
            auth.revoke(tok)
        service.store.audit(p.username, p.role, "logout", "", {})
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/selftest")
    def self_test_last(p: Principal = Depends(viewer)):
        return service.last_self_test or {"ok": None, "detail": "no self-test has run yet"}

    @app.post("/api/selftest")
    def self_test_run(body: SelfTest, p: Principal = Depends(admin)):
        return service.self_test(notify=body.notify, actor=p.username)

    @app.post("/api/console/sound")
    def console_sound(body: Sound, p: Principal = Depends(viewer)):
        """The console's audible alarm was switched on/off - muting is an auditable act."""
        service.store.audit(p.username, p.role, "console_sound_on" if body.on else "console_muted", "", {})
        return {"ok": True}

    @app.get("/api/me")
    def me(p: Principal = Depends(viewer)):
        return {"username": p.username, "role": p.role, "auth": auth.enabled}

    # ------------------------------------------------------------ status
    @app.get("/api/status")
    def status(p: Principal = Depends(viewer)):
        d = service.status()
        d["privacy"] = cfg["runtime"]["privacy"]
        return d

    @app.get("/api/cameras")
    def cameras(p: Principal = Depends(viewer)):
        return [w.status() for w in service.cameras.values()]

    @app.get("/api/cameras/{cam_id}/frame.jpg")
    def camera_frame(cam_id: str, p: Principal = Depends(viewer)):
        w = service.cameras.get(cam_id)
        if w is None or not w.latest_jpg:
            raise HTTPException(404, "no frame")
        return Response(w.latest_jpg, media_type="image/jpeg")

    @app.post("/api/cameras/{cam_id}/pause")
    def camera_pause(cam_id: str, body: Pause, p: Principal = Depends(operator)):
        st = service.pause_camera(cam_id, body.minutes * 60, p.username, p.role, body.reason)
        if st is None:
            raise HTTPException(404, "unknown camera")
        return st

    @app.post("/api/cameras/{cam_id}/resume")
    def camera_resume(cam_id: str, p: Principal = Depends(operator)):
        st = service.resume_camera(cam_id, p.username, p.role)
        if st is None:
            raise HTTPException(404, "unknown camera")
        return st

    @app.get("/api/stats")
    def stats(hours: float = Query(24.0, gt=0, le=24 * 365), p: Principal = Depends(viewer)):
        return service.store.stats(since=time.time() - hours * 3600)

    # ------------------------------------------------------------ alerts
    @app.get("/api/alerts")
    def alerts(limit_: int = Query(50, alias="limit", ge=1, le=500), status_: Optional[str] = Query(None, alias="status"),
               hazard: Optional[str] = None, camera: Optional[str] = None, p: Principal = Depends(viewer)):
        rows = service.store.list_alerts(limit_, status=status_, hazard=hazard, camera_id=camera)
        for r in rows:
            r["time_utc"] = utc_iso(r.get("wall_time"))
            for k in ("snapshot_path", "evidence_path", "clip_path"):
                r[f"has_{k.split('_')[0]}"] = bool(r.pop(k, None))
        return rows

    def _alert_or_404(alert_id: str) -> dict:
        if len(alert_id) > 40:
            raise HTTPException(404, "unknown alert")
        a = service.store.get_alert(alert_id)
        if a is None:
            raise HTTPException(404, "unknown alert")
        return a

    @app.get("/api/alerts/{alert_id}")
    def alert(alert_id: str, p: Principal = Depends(viewer)):
        a = _alert_or_404(alert_id)
        for k, out in (("snapshot_path", "snapshot_jpg_b64"), ("evidence_path", "evidence_jpg_b64")):
            data = service.store.get_media(a.pop(k)) if a.get(k) else None
            a[out] = base64.b64encode(data).decode() if data else ""
        a["has_clip"] = bool(a.pop("clip_path", None))
        a["time_utc"] = utc_iso(a.get("wall_time"))
        return a

    @app.get("/api/alerts/{alert_id}/clip.mp4")
    def alert_clip(alert_id: str, p: Principal = Depends(operator)):
        a = _alert_or_404(alert_id)
        data = service.store.get_media(a["clip_path"]) if a.get("clip_path") else None
        if not data:
            raise HTTPException(404, "no clip (yet)")
        service.store.audit(p.username, p.role, "clip_viewed", alert_id, {})
        return Response(data, media_type="video/mp4")

    @app.post("/api/alerts/{alert_id}/feedback")
    def feedback(alert_id: str, fb: Feedback, p: Principal = Depends(operator)):
        a = _alert_or_404(alert_id)
        try:
            r = service.feedback(alert_id, fb.confirmed, p.username, p.role, fb.note)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        if r is None:
            raise HTTPException(404, "unknown alert")
        prom.FEEDBACK.labels(camera=a.get("camera_id", "?"), hazard=a.get("hazard", "?"),
                             outcome="confirmed" if fb.confirmed else "dismissed").inc()
        return r

    @app.post("/api/alerts/{alert_id}/whatif")
    def what_if(alert_id: str, body: WhatIf, p: Principal = Depends(viewer)):
        _alert_or_404(alert_id)
        try:
            r = service.what_if(alert_id, body.values)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        if r is None:
            raise HTTPException(404, "unknown alert")
        return r

    @app.post("/api/alerts/{alert_id}/ask")
    def ask(alert_id: str, q: Question, p: Principal = Depends(viewer)):
        a = _alert_or_404(alert_id)
        return answer(a, q.question)

    @app.get("/api/alerts/{alert_id}/report.html", response_class=HTMLResponse)
    def report(alert_id: str, p: Principal = Depends(operator)):
        a = _alert_or_404(alert_id)
        imgs = {}
        for k in ("snapshot_path", "evidence_path"):
            data = service.store.get_media(a[k]) if a.get(k) else None
            imgs[k] = base64.b64encode(data).decode() if data else ""
        service.store.audit(p.username, p.role, "report_exported", alert_id, {})
        return HTMLResponse(render_report(a, imgs, p.username))

    @app.post("/api/alerts/{alert_id}/hold")
    def alert_hold(alert_id: str, body: Hold, p: Principal = Depends(admin)):
        _alert_or_404(alert_id)
        service.store.update_alert(alert_id, legal_hold=int(body.hold))
        service.store.audit(p.username, p.role, "legal_hold_set" if body.hold else "legal_hold_released", alert_id, {})
        return {"id": alert_id, "legal_hold": body.hold}

    @app.post("/api/alerts/{alert_id}/erase")
    def alert_erase(alert_id: str, p: Principal = Depends(admin)):
        _alert_or_404(alert_id)
        try:
            r = service.store.erase(alert_id)
        except PermissionError as e:
            raise HTTPException(409, str(e)) from e
        service.store.audit(p.username, p.role, "erased", alert_id, r)
        return r

    # ------------------------------------------------------------- admin
    @app.get("/api/audit")
    def audit(limit_: int = Query(200, alias="limit", ge=1, le=5000), p: Principal = Depends(admin)):
        return service.store.audit_log(limit_)

    @app.get("/api/audit/verify")
    def audit_verify(p: Principal = Depends(admin)):
        return service.store.verify_audit_chain()

    # ------------------------------------------------------------ devices
    @app.post("/api/sensors")
    def sensors(body: SensorReading, request: Request):
        limit(request, bucket="sensor")
        dev = auth.verify_device(request.headers.get("x-api-key"))
        if dev is None:
            raise HTTPException(401, "device key required")
        if not dev.may_report(body.zone):
            service.store.audit(dev.username, "device", "sensor_zone_denied", body.zone, {"ip": client_ip(request)})
            raise HTTPException(403, "this device is not registered for that zone")
        rec = body.model_dump(exclude_none=True)
        if len(rec) <= 1:
            raise HTTPException(422, "no sensor values")
        rec["t"] = time.time()  # server clock: devices cannot inject stale or future readings
        if not service.sensors.push(rec):
            raise HTTPException(422, "unknown zone")
        return {"ok": True}

    # ----------------------------------------------------------- live feed
    def _fanout(msg: dict) -> None:
        loop = loop_ref.get("loop")
        if loop is None:
            return
        for c in list(clients):
            loop.call_soon_threadsafe(c.offer, msg)

    class Client:
        def __init__(self, ws: WebSocket):
            self.ws = ws
            self.q: asyncio.Queue = asyncio.Queue(maxsize=64)
            self.last_frame: dict = {}

        def offer(self, msg: dict) -> None:
            if msg["type"] == "frame":
                now = time.monotonic()
                if now - self.last_frame.get(msg["camera"], 0) < stream_period:
                    return  # throttle video per client and camera
                self.last_frame[msg["camera"]] = now
                if self.q.qsize() > 8:
                    return  # slow client: drop frames, never alerts
            if self.q.full():
                try:
                    self.q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self.q.put_nowait(msg)

    allowed_origins = set(scfg.get("cors_origins") or [])

    def origin_ok(sock: WebSocket) -> bool:
        origin = sock.headers.get("origin")
        if origin is None:
            return True  # non-browser clients (scripts) send no Origin; they still need a token
        host = sock.headers.get("host", "")
        return origin in allowed_origins or origin.split("://", 1)[-1] == host

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        if not origin_ok(sock):  # cross-site WebSocket hijacking guard
            await sock.close(code=4403)
            return
        tok = sock.cookies.get(COOKIE) or sock.query_params.get("token")
        p = auth.verify(tok)
        if p is None:
            await sock.close(code=4401)
            return
        if len(clients) >= max_ws:
            await sock.close(code=4429)
            return
        await sock.accept()
        c = Client(sock)
        clients.add(c)
        try:
            await sock.send_text(json.dumps({"type": "hello", "user": p.username, "role": p.role,
                                             "cameras": [w.status() for w in service.cameras.values()],
                                             "alerts": service.store.list_alerts(30)}, default=str))
            while True:
                try:
                    # heartbeat: with no frames (camera offline, file ended) a dead client is
                    # still noticed, and shutdown is never blocked by an idle handler
                    msg = await asyncio.wait_for(c.q.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    msg = {"type": "ping"}
                if msg["type"] == "frame":
                    out = {"type": "frame", "camera": msg["camera"], "state": msg["state"],
                           "jpg": base64.b64encode(msg["jpg"]).decode()}
                else:
                    out = msg
                await sock.send_text(json.dumps(out, default=str))
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:  # noqa: BLE001 - never let one client's error bubble into the server loop
            log.exception("websocket client failed")
        finally:
            clients.discard(c)

    return app


def render_report(a: dict, imgs: dict, user: str) -> str:
    """Self-contained incident report (print to PDF from the browser)."""
    e = lambda x: html.escape(str(x if x is not None else ""))  # noqa: E731
    exp = a.get("explanation") or {}
    cues = "".join(f"<tr><td>{e(c['phrase'])}</td><td>{c['contribution']:+.2f}</td><td>{c.get('reliability', 1):.2f}</td></tr>"
                   for c in a.get("cues") or [])
    recs = "".join(f"<li>{e(r['text'])}</li>" for r in a.get("recommendations") or [])
    cfs = "".join(f"<li>{e(c['text'])}</li>" for c in (a.get("decision") or {}).get("counterfactuals", []))
    img = lambda k, cap: (f'<figure><img src="data:image/jpeg;base64,{imgs[k]}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'  # noqa: E731
                          if imgs.get(k) else "")
    t = f"{utc_iso(a.get('wall_time') or 0)} (UTC)"
    ver = a.get("verification") or {}
    prov = a.get("provenance") or {}
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Incident {e(a['id'])}</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:860px;margin:24px auto;color:#111}}h1{{font-size:22px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
figure{{display:inline-block;margin:0 12px 12px 0;width:48%}}img{{width:100%}}.k{{color:#555}}</style></head><body>
<h1>Incident report {e(a['id'])}</h1>
<p class="k">Generated {e(utc_iso(time.time()))} (UTC) by {e(user)} · Explainable Edge AI Copilot {e(prov.get('version', ''))}</p>
<p class="k"><b>Decision support only.</b> This system supplements, and does not replace, the building's certified fire
alarm system (IS 2189 / NBC 2016) or professional medical assessment.</p>
<table><tr><th>Time</th><td>{e(t)}</td><th>Camera / zone</th><td>{e(a.get('camera_id'))} / {e(a.get('zone'))}</td></tr>
<tr><th>Hazard</th><td>{e(a.get('hazard'))}</td><th>Severity / triage</th><td>{e(a.get('severity'))} / {e(a.get('triage'))}</td></tr>
<tr><th>Probability</th><td>{(a.get('prob') or 0):.2f} (peak {(a.get('peak_prob') or 0):.2f})</td><th>Status</th><td>{e(a.get('status'))} by {e(a.get('operator'))}</td></tr></table>
<h2>Explanation</h2><p>{e(exp.get('text'))}</p>
<p class="k">Grounding check: {ver.get('groundedness', 0):.0%} of {e(ver.get('n_claims'))} claims supported by the evidence.</p>
<h2>Evidence ledger</h2><table><tr><th>Cue</th><th>Contribution (log-odds)</th><th>Reliability</th></tr>{cues}</table>
<h2>Counterfactuals</h2><ul>{cfs or '<li>none</li>'}</ul>
<h2>Recommended actions</h2><ol>{recs}</ol>
<h2>Evidence</h2>{img('snapshot_path', 'Snapshot (anonymised)')}{img('evidence_path', 'What the model relied on')}
<h2>Operator note</h2><p>{e(a.get('operator_note'))}</p>
<p class="k">Provenance: config {e(prov.get('config_sha256'))}, models {e(json.dumps(prov.get('models', {})))}</p>
</body></html>"""


def build(args) -> tuple:
    cfg = load_config_dict(args.config)
    if args.privacy:
        cfg["runtime"]["privacy"] = args.privacy
    a = cfg["server"]["auth"]
    host = args.host or cfg["server"]["host"]
    if args.demo:
        # local demo without login: loopback only, never on a public bind
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise SystemExit("--demo is only allowed on a loopback address")
        a["enabled"] = False
    elif a["enabled"] and not a["users"]:
        raise SystemExit("No users configured (server.auth.users). Create one with: python -m edgecopilot.security hash")
    secret = load_secret(a.get("secret_env", "EDGECOPILOT_SECRET"), os.path.join(ROOT, "data", ".secret"))
    auth = AuthManager(a["users"], secret, a["token_ttl_s"], a["sensor_api_keys"], enabled=a["enabled"],
                       sensor_devices=a.get("sensor_devices", []))
    service = EdgeService(cfg, root=ROOT, audit_key=secret)
    cams = cfg["cameras"]
    if args.source is not None:
        cams = [{"id": "cam1", "source": args.source, "zone": args.zone, "stride": 1, "realtime": True,
                 "loop": not args.no_loop, "hazards": {}}]
    if not cams:
        raise SystemExit("No cameras: add a `cameras:` list to the config or pass --source")
    for c in cams:
        if not c.get("enabled", True):
            continue
        src = str(c["source"])
        csv = os.path.splitext(src)[0] + "_sensors.csv" if not src.startswith(("rtsp", "http")) else None
        service.add_camera(c, sensors_csv=csv if csv and os.path.exists(csv) else None)
    return cfg, auth, service


def main():
    ap = argparse.ArgumentParser(description="Explainable Edge AI Copilot server")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "campus.yaml"))
    ap.add_argument("--source", default=None, help="single camera: video file, webcam index or RTSP URL")
    ap.add_argument("--zone", default="Corridor B")
    ap.add_argument("--privacy", choices=["raw", "blur", "skeleton"], default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--demo", action="store_true", help="loopback-only demo mode without login")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg, auth, service = build(args)
    host = args.host or cfg["server"]["host"]
    if host not in ("127.0.0.1", "localhost", "::1") and not auth.enabled:
        raise SystemExit("refusing to serve without authentication on a non-loopback address")
    import uvicorn
    uvicorn.run(create_app(service, auth, cfg), host=host, port=args.port or cfg["server"]["port"],
                log_level="warning", proxy_headers=True, server_header=False,
                timeout_graceful_shutdown=5,  # SIGTERM (systemd/docker stop) never hangs on open connections
                # trust X-Forwarded-For only from the TLS proxy, so lockout/rate limits see the real client
                forwarded_allow_ips=os.environ.get("EDGECOPILOT_TRUSTED_PROXIES", "127.0.0.1"))


if __name__ == "__main__":
    main()
