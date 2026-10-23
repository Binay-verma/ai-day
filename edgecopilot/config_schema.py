"""Typed, validated configuration (fail fast, fail loudly).

A safety system must not start with a silently broken config: a typo in a
playbook condition, a cue phrase that the verifier cannot bind, a threshold
of 6 instead of 0.6, or a camera pointing at a zone that does not exist would
otherwise surface only during an emergency.  ``load_validated`` parses the
YAML into pydantic models and runs cross-field checks; the server and CLI
refuse to start on any error and print every problem at once.

    python -m edgecopilot.config_schema configs/campus.yaml   # lint a config
"""
from __future__ import annotations

import ast
import os
import re
import sys
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuntimeCfg(_Strict):
    threads: int = Field(2, ge=1, le=64)
    privacy: Literal["raw", "blur", "skeleton"] = "blur"
    motion_gate: bool = True
    gate_keepalive: int = Field(8, ge=1, le=300)
    fire_every: int = Field(2, ge=1, le=100)
    max_width: int = Field(960, ge=160, le=7680)
    scene_cut: float = Field(0.75, gt=0, le=1)
    exposure_jump: float = Field(0.45, gt=0, le=1)
    realert_cooldown_s: float = Field(30, ge=0, le=3600)
    track_high_conf: float = Field(0.35, gt=0, lt=1)


class ModelCfg(_Strict):
    path: str
    imgsz: int = Field(640, ge=160, le=1920)
    conf: float = Field(0.25, gt=0, lt=1)
    iou: float = Field(0.45, gt=0, lt=1)
    classes: Optional[List[str]] = None
    sha256: Optional[str] = Field(None, pattern=r"^[0-9a-f]{64}$", description="pin: refuse to load any other file")


class ModelsCfg(_Strict):
    pose: ModelCfg
    fire_smoke: ModelCfg


class CueCfg(_Strict):
    weight: float
    ref: float
    lo: float
    hi: float
    unit: str = ""
    phrase: str
    source: Literal["vision", "sensor", "context"] = "vision"
    keywords: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _range(self):
        if not self.lo < self.hi:
            raise ValueError(f"lo ({self.lo}) must be < hi ({self.hi})")
        if not self.lo <= self.ref <= self.hi:
            raise ValueError(f"ref ({self.ref}) must lie in [lo, hi] = [{self.lo}, {self.hi}]")
        try:
            self.phrase.format(v=1.0, name="x", unit=self.unit)
        except (KeyError, ValueError, IndexError) as e:
            raise ValueError(f"phrase {self.phrase!r} cannot be formatted: {e}") from e
        return self


class HazardCfg(_Strict):
    bias: float
    threshold: float = Field(0.6, gt=0, lt=1)
    cues: Dict[str, CueCfg]
    alpha: Optional[float] = Field(None, gt=0, lt=1, description="conformal false-alarm budget")
    min_threshold: float = Field(0.2, gt=0, lt=1, description="floor for calibrated alert lines")
    max_threshold: float = Field(0.9, gt=0, lt=1, description="ceiling for calibrated alert lines: operator "
                                 "feedback can never desensitise a camera beyond this")

    @model_validator(mode="after")
    def _threshold_order(self):
        if not self.min_threshold <= self.threshold <= self.max_threshold:
            raise ValueError(f"need min_threshold <= threshold <= max_threshold, got "
                             f"{self.min_threshold} / {self.threshold} / {self.max_threshold}")
        return self


class HysteresisCfg(_Strict):
    alpha: float = Field(0.6, gt=0, le=1)
    on_frames: int = Field(2, ge=1, le=100)
    off_frames: int = Field(30, ge=1, le=1000)
    off_margin: float = Field(0.15, ge=0, lt=1)


class FeedbackCfg(_Strict):
    lr: float = Field(0.5, gt=0, le=5)
    max_shift: float = Field(2.0, ge=0, le=10)


class ZoneCfg(_Strict):
    capacity: int = Field(50, ge=1)
    exits: List[str] = Field(default_factory=list)
    aed: str = ""
    extinguisher: str = ""
    first_aid: str = ""
    assembly_point: str = ""
    warden: str = ""


class PlaybookStep(_Strict):
    do: str
    priority: int = Field(2, ge=1, le=5)
    when: Optional[str] = None
    escalate: bool = False


class TriageCfg(_Strict):
    min_margin: float = 1.0
    max_conflict: float = 0.35
    strong: float = 0.5


class CameraCfg(_Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,40}$")
    source: str
    zone: str
    stride: int = Field(1, ge=1, le=30)
    privacy: Optional[Literal["raw", "blur", "skeleton"]] = None
    enabled: bool = True
    hazards: Dict[str, bool] = Field(default_factory=dict)
    realtime: bool = True
    loop: bool = False


class UserCfg(_Strict):
    username: str = Field(pattern=r"^[a-z0-9_.\-]{2,32}$")
    role: Literal["viewer", "operator", "admin"]
    password_hash: str  # scrypt$<salt hex>$<hash hex>, see `python -m edgecopilot.security hash`


class SensorDeviceCfg(_Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_.\-]{1,40}$")
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    zones: List[str] = Field(min_length=1)


class AuthCfg(_Strict):
    enabled: bool = True
    users: List[UserCfg] = Field(default_factory=list)
    token_ttl_s: int = Field(8 * 3600, ge=60, le=7 * 24 * 3600)
    sensor_api_keys: List[str] = Field(default_factory=list,
                                       description="legacy: digests of keys allowed to post for ANY zone (avoid)")
    sensor_devices: List[SensorDeviceCfg] = Field(default_factory=list,
                                                    description="device key digest bound to the zone(s) it may report for")
    secret_env: str = "EDGECOPILOT_SECRET"  # noqa: S105 - name of an env var, not a secret


class ServerCfg(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    cors_origins: List[str] = Field(default_factory=list)
    max_ws_clients: int = Field(20, ge=1, le=500)
    stream_fps: float = Field(5.0, gt=0, le=30)
    rate_limit_per_min: int = Field(240, ge=10)
    auth: AuthCfg = Field(default_factory=lambda: AuthCfg())


class StorageCfg(_Strict):
    db_path: str = "data/edgecopilot.db"
    media_dir: str = "data/media"
    retention_days: int = Field(30, ge=1, le=3650)
    encrypt_media: bool = True
    key_env: str = "EDGECOPILOT_MEDIA_KEY"
    clip_pre_s: float = Field(5.0, ge=0, le=60)
    clip_post_s: float = Field(10.0, ge=0, le=120)


class WebhookCfg(_Strict):
    url: str
    kind: Literal["generic", "slack", "teams"] = "generic"
    min_severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"


class EmailCfg(_Strict):
    host: str
    port: int = 587
    user_env: str = "SMTP_USER"
    password_env: str = "SMTP_PASSWORD"  # noqa: S105 - name of an env var, not a secret
    sender: str
    to: List[str]
    starttls: bool = True
    min_severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "HIGH"


class EscalationLevel(_Strict):
    after_s: float = Field(ge=0)
    channels: List[str]


class NotificationsCfg(_Strict):
    webhooks: Dict[str, WebhookCfg] = Field(default_factory=dict)
    email: Optional[EmailCfg] = None
    mqtt_topic: Optional[str] = None
    escalation: List[EscalationLevel] = Field(default_factory=list)
    max_retries: int = Field(5, ge=0, le=20)


class MqttCfg(_Strict):
    host: str = "localhost"
    port: int = Field(1883, ge=1, le=65535)
    sensor_topic: str = "campus/+/sensors"
    username_env: Optional[str] = None
    password_env: Optional[str] = None
    tls: bool = False


class Config(_Strict):
    runtime: RuntimeCfg = Field(default_factory=lambda: RuntimeCfg())
    models: ModelsCfg
    hazards: Dict[str, HazardCfg]
    hysteresis: HysteresisCfg = Field(default_factory=lambda: HysteresisCfg())
    feedback: FeedbackCfg = Field(default_factory=lambda: FeedbackCfg())
    contacts: Dict[str, str] = Field(default_factory=dict)
    zones: Dict[str, ZoneCfg]
    playbooks: Dict[str, List[PlaybookStep]] = Field(default_factory=dict)
    triage: TriageCfg = Field(default_factory=lambda: TriageCfg())
    cameras: List[CameraCfg] = Field(default_factory=list)
    server: ServerCfg = Field(default_factory=lambda: ServerCfg())
    storage: StorageCfg = Field(default_factory=lambda: StorageCfg())
    notifications: NotificationsCfg = Field(default_factory=lambda: NotificationsCfg())
    mqtt: Optional[MqttCfg] = None

    @field_validator("hazards")
    @classmethod
    def _known_hazards(cls, v):
        unknown = set(v) - {"fall", "fire_smoke", "crowd"}
        if unknown:
            raise ValueError(f"unknown hazard(s) {sorted(unknown)}; expected fall, fire_smoke, crowd")
        return v

    @model_validator(mode="after")
    def _cross_checks(self):
        errors: List[str] = []
        # 1) every cue phrase binds to its own keyword and to no other cue's keyword
        for hz, h in self.hazards.items():
            for name, c in h.cues.items():
                kws = c.keywords or [w for w in name.split("_") if len(w) > 2]
                phrase = c.phrase.format(v=1.0, name=name, unit=c.unit).lower()
                if not any(re.search(rf"\b{re.escape(k.lower())}", phrase) for k in kws):
                    errors.append(f"hazards.{hz}.cues.{name}: phrase does not contain any of its keywords {kws}")
                for other, oc in h.cues.items():
                    if other == name:
                        continue
                    okws = oc.keywords or [w for w in other.split("_") if len(w) > 2]
                    clash = [k for k in okws if re.search(rf"\b{re.escape(k.lower())}", phrase)]
                    if clash:
                        errors.append(f"hazards.{hz}.cues.{name}: phrase matches keyword(s) {clash} of cue '{other}'")
                    oname = other.replace("_", " ")
                    if any(re.search(rf"\b{re.escape(k.lower())}\b", oname) for k in kws):
                        errors.append(f"hazards.{hz}.cues.{name}: a keyword also matches the name of cue '{other}'")
        # 2) playbook conditions are well-formed and only use known names
        from .copilot.recommender import safe_eval
        for hz, steps in self.playbooks.items():
            if hz not in self.hazards:
                errors.append(f"playbooks.{hz}: no such hazard")
                continue
            names = set(self.hazards[hz].cues) | {"prob"}
            for i, st in enumerate(steps):
                if not st.when:
                    continue
                try:
                    tree = ast.parse(st.when, mode="eval")
                except SyntaxError as e:
                    errors.append(f"playbooks.{hz}[{i}].when: syntax error: {e.msg}")
                    continue
                used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                if used - names:
                    errors.append(f"playbooks.{hz}[{i}].when uses unknown name(s) {sorted(used - names)}")
                    continue
                try:
                    safe_eval(st.when, {n: 0.0 for n in names})
                except ValueError as e:
                    errors.append(f"playbooks.{hz}[{i}].when: {e}")
        # 3) cameras reference existing zones, unique ids
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            errors.append("cameras: duplicate camera ids")
        for c in self.cameras:
            if c.zone not in self.zones:
                errors.append(f"cameras[{c.id}].zone '{c.zone}' is not defined in zones")
            bad = set(c.hazards) - set(self.hazards)
            if bad:
                errors.append(f"cameras[{c.id}].hazards: unknown {sorted(bad)}")
        # 4) escalation channels exist
        chans = set(self.notifications.webhooks) | ({"email"} if self.notifications.email else set()) | \
            ({"mqtt"} if self.notifications.mqtt_topic else set())
        for i, lvl in enumerate(self.notifications.escalation):
            bad = set(lvl.channels) - chans
            if bad:
                errors.append(f"notifications.escalation[{i}]: unknown channel(s) {sorted(bad)}")
        # 5) sensor devices may only report for existing zones
        for d in self.server.auth.sensor_devices:
            bad = set(d.zones) - set(self.zones)
            if bad:
                errors.append(f"server.auth.sensor_devices[{d.id}]: unknown zone(s) {sorted(bad)}")
        # 6) auth on a non-loopback bind
        if self.server.host not in ("127.0.0.1", "localhost", "::1") and not self.server.auth.enabled:
            errors.append("server: authentication must be enabled when binding to a non-loopback address")
        if errors:
            raise ValueError("; ".join(errors))
        return self


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_raw(path: str, _depth: int = 0) -> dict:
    """YAML with optional ``extends: base.yaml`` overlay (site config on top of the shared model config)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("extends", None)
    if base:
        if _depth > 3:
            raise ValueError("config 'extends' chain too deep")
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(os.path.abspath(path)), base)
        raw = _deep_merge(load_raw(base_path, _depth + 1), raw)
    return raw


def load_validated(path: str) -> Config:
    return Config.model_validate(load_raw(path))


def load_config_dict(path: str) -> dict:
    """Validated config as a plain dict (what the pipeline consumes), defaults filled in."""
    return load_validated(path).model_dump(exclude_none=False)


def check_models(cfg: Config, root: str = ".") -> List[str]:
    """Warnings for missing model files (the copilot degrades to cue-only mode)."""
    warn = []
    for name in ("pose", "fire_smoke"):
        p = getattr(cfg.models, name).path
        p = p if os.path.isabs(p) else os.path.join(root, p)
        if not os.path.exists(p):
            warn.append(f"model '{name}' not found at {p}: running without it")
    return warn


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    path = argv[0] if argv else "configs/campus.yaml"
    try:
        cfg = load_validated(path)
    except ValidationError as e:
        print(f"INVALID {path}:")
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            print(f"  - {loc}: {err['msg']}")
        return 1
    for w in check_models(cfg, os.path.dirname(os.path.dirname(os.path.abspath(path)))):
        print("WARNING", w)
    print(f"OK {path}: {len(cfg.hazards)} hazards, {sum(len(h.cues) for h in cfg.hazards.values())} cues, "
          f"{len(cfg.zones)} zones, {len(cfg.cameras)} cameras")
    return 0


if __name__ == "__main__":
    sys.exit(main())
