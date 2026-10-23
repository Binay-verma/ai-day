"""Durable incident store, tamper-evident audit trail and encrypted media.

* **SQLite (WAL)** keeps alerts and operator actions across restarts.
* **Audit log is hash-chained**: every row stores SHA-256(previous hash +
  row content), so deleting or editing any past entry breaks the chain and
  ``verify_audit_chain()`` reports where.  Who confirmed or dismissed an
  alert, who logged in, who exported what - all recorded.
* **Media (snapshots, evidence maps, clips) are encrypted at rest** with
  Fernet (AES-128-CBC + HMAC-SHA256); the key comes from an environment
  variable or a 0600 key file.  Media are already privacy-filtered (blurred
  heads / skeletons) before they are written.
* **Retention**: ``purge(retention_days)`` deletes media and personal fields
  of old incidents (data minimisation, India DPDP Act 2023) while keeping the
  anonymous statistics needed for safety reviews.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:  # pragma: no cover
    Fernet = None
    InvalidToken = Exception

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
  id TEXT PRIMARY KEY, camera_id TEXT, zone TEXT, hazard TEXT, subject TEXT,
  t_video REAL, wall_time REAL, prob REAL, peak_prob REAL, severity TEXT, triage TEXT,
  status TEXT, operator TEXT, operator_note TEXT, ack_time REAL,
  explanation TEXT, decision TEXT, verification TEXT, recommendations TEXT, cues TEXT,
  provenance TEXT, snapshot_path TEXT, evidence_path TEXT, clip_path TEXT, purged INTEGER DEFAULT 0,
  legal_hold INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_alerts_time ON alerts(wall_time);
CREATE INDEX IF NOT EXISTS ix_alerts_status ON alerts(status);
CREATE TABLE IF NOT EXISTS audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, actor TEXT, role TEXT, action TEXT,
  target TEXT, detail TEXT, prev_hash TEXT, hash TEXT
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""
JSON_FIELDS = ("explanation", "decision", "verification", "recommendations", "cues", "provenance", "triage")


def _load_key(env: str, path: str) -> Optional[bytes]:
    if Fernet is None:
        return None
    val = os.environ.get(env)
    if val:
        return val.encode()
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read().strip()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    key = Fernet.generate_key()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


class Store:
    def __init__(self, db_path: str = "data/edgecopilot.db", media_dir: str = "data/media",
                 encrypt_media: bool = True, key_env: str = "EDGECOPILOT_MEDIA_KEY"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(media_dir, exist_ok=True)
        self.db_path = db_path
        self.media_dir = media_dir
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.fernet = None
        if encrypt_media:
            key = _load_key(key_env, os.path.join(os.path.dirname(db_path) or ".", ".media_key"))
            if key is None:
                raise RuntimeError("encrypt_media=true but the 'cryptography' package is not installed")
            self.fernet = Fernet(key)

    def _migrate(self) -> None:
        """Forward-only schema migrations for databases created by older versions."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(alerts)")}
        if "legal_hold" not in cols:
            self.conn.execute("ALTER TABLE alerts ADD COLUMN legal_hold INTEGER DEFAULT 0")
        self.conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('schema_version', '2')")

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    # --------------------------------------------------------------- media
    def put_media(self, name: str, data: bytes) -> str:
        safe = "".join(ch for ch in name if ch.isalnum() or ch in "._-")[:120]
        path = os.path.join(self.media_dir, safe + (".enc" if self.fernet else ""))
        blob = self.fernet.encrypt(data) if self.fernet else data
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)
        return path

    def get_media(self, path: str) -> Optional[bytes]:
        real = os.path.realpath(path)
        if not real.startswith(os.path.realpath(self.media_dir) + os.sep) or not os.path.exists(real):
            return None  # never serve anything outside the media directory
        with open(real, "rb") as f:
            blob = f.read()
        if self.fernet and real.endswith(".enc"):
            try:
                return self.fernet.decrypt(blob)
            except InvalidToken:
                return None
        return blob

    # -------------------------------------------------------------- alerts
    def save_alert(self, rec: Dict[str, Any]) -> None:
        cols = ["id", "camera_id", "zone", "hazard", "subject", "t_video", "wall_time", "prob", "peak_prob",
                "severity", "triage", "status", "operator", "operator_note", "ack_time", "explanation", "decision",
                "verification", "recommendations", "cues", "provenance", "snapshot_path", "evidence_path", "clip_path"]
        vals = []
        for c in cols:
            v = rec.get(c)
            if c in JSON_FIELDS and v is not None and not isinstance(v, str):
                v = json.dumps(v, default=float)
            vals.append(v)
        # column names come from the fixed list above, values are bound parameters
        q = f"INSERT OR REPLACE INTO alerts ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"  # noqa: S608  # nosec B608
        with self.lock:
            self.conn.execute(q, vals)

    def update_alert(self, alert_id: str, **fields) -> bool:
        if not fields:
            return False
        allowed = {"status", "operator", "operator_note", "ack_time", "peak_prob", "clip_path", "legal_hold"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"cannot update {bad}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self.lock:
            # keys were checked against the `allowed` whitelist above; values are bound
            cur = self.conn.execute(f"UPDATE alerts SET {sets} WHERE id = ?", [*fields.values(), alert_id])  # noqa: S608  # nosec B608
            return cur.rowcount > 0

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
            row = cur.fetchone()
            names = [d[0] for d in cur.description]
        return self._row(row, names) if row else None

    def list_alerts(self, limit: int = 100, since: Optional[float] = None, status: Optional[str] = None,
                    hazard: Optional[str] = None, camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM alerts WHERE 1=1"
        args: List[Any] = []
        for col, val in (("status", status), ("hazard", hazard), ("camera_id", camera_id)):
            if val:
                q += f" AND {col} = ?"
                args.append(val)
        if since is not None:
            q += " AND wall_time >= ?"
            args.append(since)
        q += " ORDER BY wall_time DESC LIMIT ?"
        args.append(int(max(1, min(limit, 5000))))
        with self.lock:
            cur = self.conn.execute(q, args)
            rows = cur.fetchall()
            names = [d[0] for d in cur.description]
        return [self._row(r, names) for r in rows]

    def stats(self, since: Optional[float] = None) -> Dict[str, Any]:
        q = "SELECT hazard, status, COUNT(*) FROM alerts"
        args: List[Any] = []
        if since is not None:
            q += " WHERE wall_time >= ?"
            args.append(since)
        q += " GROUP BY hazard, status"
        with self.lock:
            rows = self.conn.execute(q, args).fetchall()
        out: Dict[str, Dict[str, Any]] = {}
        for hz, st, n in rows:
            out.setdefault(hz, {})[st] = n
        # precision as judged by operators: confirmed / (confirmed + dismissed)
        for d in out.values():
            c, x = d.get("confirmed", 0), d.get("dismissed", 0)
            d["operator_precision"] = round(c / (c + x), 3) if c + x else None
        return out

    @staticmethod
    def _row(row, names) -> Dict[str, Any]:
        d = dict(zip(names, row))
        for k in JSON_FIELDS:
            if isinstance(d.get(k), str):
                try:
                    d[k] = json.loads(d[k])
                except json.JSONDecodeError:
                    pass
        return d

    def erase(self, alert_id: str) -> Dict[str, int]:
        """Data-subject erasure: delete an incident's media and personal fields now."""
        with self.lock:
            row = self.conn.execute("SELECT snapshot_path, evidence_path, clip_path, legal_hold FROM alerts WHERE id = ?",
                                    (alert_id,)).fetchone()
        if row is None:
            return {"found": 0, "files": 0}
        if row[3]:
            raise PermissionError("incident is under legal hold")
        files = 0
        for p in row[:3]:
            if p and os.path.exists(p) and os.path.realpath(p).startswith(os.path.realpath(self.media_dir)):
                os.remove(p)
                files += 1
        with self.lock:
            self.conn.execute("UPDATE alerts SET snapshot_path = NULL, evidence_path = NULL, clip_path = NULL, "
                              "operator_note = NULL, subject = 'redacted', purged = 1 WHERE id = ?", (alert_id,))
        return {"found": 1, "files": files}

    # --------------------------------------------------------------- audit
    def audit(self, actor: str, role: str, action: str, target: str = "", detail: Optional[dict] = None) -> str:
        with self.lock:
            row = self.conn.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else "0" * 64
            ts = time.time()
            det = json.dumps(detail or {}, sort_keys=True, default=str)
            h = hashlib.sha256(f"{prev}|{ts!r}|{actor}|{role}|{action}|{target}|{det}".encode()).hexdigest()
            self.conn.execute("INSERT INTO audit (ts, actor, role, action, target, detail, prev_hash, hash) "
                              "VALUES (?,?,?,?,?,?,?,?)", (ts, actor, role, action, target, det, prev, h))
            return h

    def audit_log(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (int(min(limit, 10000)),))
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]

    def verify_audit_chain(self) -> Dict[str, Any]:
        with self.lock:
            rows = self.conn.execute("SELECT seq, ts, actor, role, action, target, detail, prev_hash, hash "
                                     "FROM audit ORDER BY seq").fetchall()
        prev = "0" * 64
        for seq, ts, actor, role, action, target, det, p, h in rows:
            exp = hashlib.sha256(f"{prev}|{ts!r}|{actor}|{role}|{action}|{target}|{det}".encode()).hexdigest()
            if p != prev or h != exp:
                return {"ok": False, "broken_at_seq": seq, "entries": len(rows)}
            prev = h
        return {"ok": True, "entries": len(rows), "head": prev}

    # ----------------------------------------------------------- retention
    def purge(self, retention_days: float, now: Optional[float] = None) -> Dict[str, int]:
        """Delete media + free-text personal fields of incidents older than the retention period."""
        now = now or time.time()
        cutoff = now - retention_days * 86400
        with self.lock:
            rows = self.conn.execute("SELECT id, snapshot_path, evidence_path, clip_path FROM alerts "
                                     "WHERE wall_time < ? AND purged = 0 AND legal_hold = 0", (cutoff,)).fetchall()
        files: int = 0
        for _id, *paths in rows:
            for p in paths:
                if p and os.path.exists(p) and os.path.realpath(p).startswith(os.path.realpath(self.media_dir)):
                    os.remove(p)
                    files += 1
        with self.lock:
            self.conn.execute("UPDATE alerts SET snapshot_path = NULL, evidence_path = NULL, clip_path = NULL, "
                              "operator_note = NULL, subject = 'redacted', purged = 1 "
                              "WHERE wall_time < ? AND purged = 0 AND legal_hold = 0", (cutoff,))
            held = {p for r in self.conn.execute("SELECT snapshot_path, evidence_path, clip_path FROM alerts "
                                                 "WHERE legal_hold = 1").fetchall() for p in r if p}
        # stray media files (e.g. from a crash) older than the cutoff, except media under legal hold
        held_real = {os.path.realpath(p) for p in held}
        for fn in os.listdir(self.media_dir):
            p = os.path.join(self.media_dir, fn)
            if os.path.isfile(p) and os.path.getmtime(p) < cutoff and os.path.realpath(p) not in held_real:
                os.remove(p)
                files += 1
        if rows or files:
            self.audit("system", "admin", "retention_purge", "", {"incidents": len(rows), "files": files,
                                                                  "retention_days": retention_days})
        return {"incidents": len(rows), "files": files}
