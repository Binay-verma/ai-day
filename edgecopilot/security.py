"""Authentication, role-based access control and rate limiting for the operator API.

* Users and roles live in the config (``server.auth.users``); passwords are
  stored only as scrypt hashes (``python -m edgecopilot.security hash``).
* Sessions are short-lived HS256 JWTs signed with a server secret taken from
  an environment variable (or generated once and kept in ``data/.secret``
  with 0600 permissions).  Browsers get them in an HttpOnly, SameSite=Strict
  cookie; scripts can send ``Authorization: Bearer <token>``.
* Roles are ordered viewer < operator < admin: viewers watch, operators
  confirm/dismiss/ask, admins manage cameras, calibration and exports.
* IoT sensor nodes authenticate with per-device API keys (``X-Api-Key``);
  only SHA-256 digests of the keys are stored.
* A token-bucket limiter caps requests per client, and repeated login failures
  lock the account/IP pair for a few minutes (brute-force protection).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import jwt

ROLES = {"viewer": 0, "operator": 1, "admin": 2}
SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)


# ------------------------------------------------------------------ passwords
def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if len(password) < 10:
        raise ValueError("password must be at least 10 characters")
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, **SCRYPT)
    return f"scrypt${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_hex, hash_hex = stored.split("$")
        if algo != "scrypt":
            return False
        h = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **SCRYPT)
        return hmac.compare_digest(h.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# --------------------------------------------------------------------- tokens
def load_secret(env: str = "EDGECOPILOT_SECRET", path: str = "data/.secret") -> bytes:
    """Server signing secret: env var first, else a persisted random secret."""
    val = os.environ.get(env)
    if val:
        if len(val) < 32:
            raise ValueError(f"{env} must be at least 32 characters")
        return val.encode()
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    s = secrets.token_bytes(48)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(s)
    return s


@dataclass
class Principal:
    username: str
    role: str
    kind: str = "user"  # user | device

    def can(self, role: str) -> bool:
        return ROLES.get(self.role, -1) >= ROLES[role]


class AuthManager:
    def __init__(self, users: Iterable[dict], secret: bytes, ttl_s: int = 8 * 3600,
                 sensor_key_digests: Iterable[str] = (), enabled: bool = True,
                 max_failures: int = 5, lockout_s: int = 300):
        self.users = {u["username"]: u for u in users}
        self.secret = secret
        self.ttl_s = ttl_s
        self.sensor_keys = {d.lower() for d in sensor_key_digests}
        self.enabled = enabled
        self.max_failures = max_failures
        self.lockout_s = lockout_s
        self._fail: Dict[Tuple[str, str], list] = {}
        self._revoked: Dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- login
    def locked(self, username: str, ip: str) -> bool:
        with self._lock:
            hist = [t for t in self._fail.get((username, ip), []) if time.time() - t < self.lockout_s]
            self._fail[(username, ip)] = hist
            return len(hist) >= self.max_failures

    def login(self, username: str, password: str, ip: str = "?") -> Optional[str]:
        if self.locked(username, ip):
            return None
        u = self.users.get(username)
        # always run one scrypt so that unknown users take the same time
        ok = verify_password(password, u["password_hash"] if u else "scrypt$00$00")
        if not (u and ok):
            with self._lock:
                self._fail.setdefault((username, ip), []).append(time.time())
            return None
        with self._lock:
            self._fail.pop((username, ip), None)
        return self.issue(username, u["role"])

    def issue(self, username: str, role: str) -> str:
        now = int(time.time())
        payload = {"sub": username, "role": role, "iat": now, "exp": now + self.ttl_s, "jti": uuid.uuid4().hex}
        return jwt.encode(payload, self.secret, algorithm="HS256")

    def revoke(self, token: str) -> None:
        try:
            p = jwt.decode(token, self.secret, algorithms=["HS256"])
        except jwt.PyJWTError:
            return
        with self._lock:
            self._revoked[p["jti"]] = p["exp"]
            now = time.time()
            for k in [k for k, exp in self._revoked.items() if exp < now]:
                del self._revoked[k]

    def verify(self, token: Optional[str]) -> Optional[Principal]:
        if not self.enabled:
            return Principal("anonymous", "admin")
        if not token:
            return None
        try:
            p = jwt.decode(token, self.secret, algorithms=["HS256"], options={"require": ["exp", "sub", "jti"]})
        except jwt.PyJWTError:
            return None
        if p.get("jti") in self._revoked or p.get("role") not in ROLES:
            return None
        u = self.users.get(p["sub"])
        if u is None or u["role"] != p["role"]:  # user removed or demoted since the token was issued
            return None
        return Principal(p["sub"], p["role"])

    def verify_device(self, api_key: Optional[str]) -> Optional[Principal]:
        if not self.enabled:
            return Principal("device", "viewer", "device")
        if not api_key:
            return None
        d = key_digest(api_key)
        for k in self.sensor_keys:
            if hmac.compare_digest(d, k):
                return Principal(f"device:{d[:8]}", "viewer", "device")
        return None


class RateLimiter:
    """Token bucket per key (client IP + route class); bounded memory."""

    def __init__(self, per_min: int = 240, burst: Optional[int] = None, max_keys: int = 10000):
        self.rate = per_min / 60.0
        self.burst = burst or max(10, per_min // 4)
        self.max_keys = max_keys
        self.buckets: "OrderedDict[str, Tuple[float, float]]" = OrderedDict()
        self.lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.pop(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            ok = tokens >= cost
            if ok:
                tokens -= cost
            self.buckets[key] = (tokens, now)
            while len(self.buckets) > self.max_keys:
                self.buckets.popitem(last=False)
            return ok


def main(argv=None) -> int:
    import getpass
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] not in ("hash", "keygen"):
        print("usage: python -m edgecopilot.security hash | keygen")
        return 2
    if argv[0] == "hash":
        pw = getpass.getpass("password: ")
        if pw != getpass.getpass("repeat:   "):
            print("passwords differ")
            return 1
        print(hash_password(pw))
    else:
        key = secrets.token_urlsafe(32)
        print("device key (put on the ESP32, shown once):", key)
        print("digest (put in server.auth.sensor_api_keys):", key_digest(key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
