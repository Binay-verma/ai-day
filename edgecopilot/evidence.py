"""Signed incident evidence bundles (idea ported from the team's Argus project, re-implemented here).

One zip per alert that an investigator, insurer or the police can take away and check later
without trusting whoever handed it over:

* ``alert.json``   - the full record: explanation, every ledger term, the decision, the verifier result
* ``report.html``  - the printable incident report
* ``snapshot.jpg`` / ``evidence.jpg`` / ``clip.mp4`` - the privacy-filtered media, when present
* ``audit.json``   - every audit-trail entry about this alert (who saw, confirmed, exported it) and the
  audit-chain head at export time
* ``MANIFEST.json`` - SHA-256 and size of every file above, who exported it and when
* ``SIGNATURE``    - HMAC-SHA256 of the manifest with the site's secret

:func:`verify_bundle` recomputes every hash and the signature and reports exactly what does not match
(a changed, missing or extra file, or a forged manifest). ``python scripts/verify_evidence.py`` and
``POST /api/evidence/verify`` wrap it.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import time
import zipfile
from typing import Any, Callable, Dict, List, Optional

FORMAT = "edgecopilot-evidence/1"
MEDIA = (("snapshot_path", "snapshot.jpg"), ("evidence_path", "evidence.jpg"), ("clip_path", "clip.mp4"))
MAX_BUNDLE = 200 * 1024 * 1024


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def sign(manifest: Dict[str, Any], key: bytes) -> str:
    return hmac.new(key, _canon(manifest), hashlib.sha256).hexdigest()


def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def build_bundle(alert: Dict[str, Any], *, key: bytes, actor: str, get_media: Callable[[str], Optional[bytes]],
                 audit_rows: List[Dict[str, Any]], audit_head: str, report_html: str,
                 now: Optional[float] = None) -> bytes:
    """Return the zip bytes. ``audit_rows`` are this alert's audit entries (any order)."""
    if not key:
        raise ValueError("an evidence bundle needs the site secret to be signed")
    now = time.time() if now is None else now
    files: Dict[str, bytes] = {
        "alert.json": json.dumps(alert, indent=1, sort_keys=True, default=str).encode(),
        "report.html": report_html.encode(),
    }
    for field, name in MEDIA:
        data = get_media(alert[field]) if alert.get(field) else None
        if data:
            files[name] = data
    rows = []
    for r in sorted(audit_rows, key=lambda x: x.get("seq") or 0):
        d = r.get("detail")
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except ValueError:
                pass
        rows.append({k: r.get(k) for k in ("seq", "ts", "actor", "role", "action", "target", "prev_hash", "hash")} | {"detail": d})
    files["audit.json"] = json.dumps({"entries": rows, "chain_head_at_export": audit_head}, indent=1, default=str).encode()
    manifest = {
        "format": FORMAT, "alert_id": alert.get("id"), "hazard": alert.get("hazard"), "zone": alert.get("zone"),
        "camera_id": alert.get("camera_id"), "exported_by": actor, "exported_utc": _utc(now),
        "audit_chain_head": audit_head,
        "files": [{"name": n, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)} for n, b in sorted(files.items())],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in sorted(files.items()):
            z.writestr(n, b)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=1, sort_keys=True))
        z.writestr("SIGNATURE", sign(manifest, key) + "\n")
    return buf.getvalue()


def verify_bundle(data: bytes, key: Optional[bytes]) -> Dict[str, Any]:
    """Check a bundle. ``ok`` is True only if the signature and every file hash match exactly."""
    problems: List[str] = []
    if len(data) > MAX_BUNDLE:
        return {"ok": False, "problems": ["bundle too large"]}
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return {"ok": False, "problems": ["not a zip file"]}
    with z:
        names = set(z.namelist())
        if "MANIFEST.json" not in names or "SIGNATURE" not in names:
            return {"ok": False, "problems": ["MANIFEST.json or SIGNATURE missing"]}
        try:
            manifest = json.loads(z.read("MANIFEST.json"))
        except ValueError:
            return {"ok": False, "problems": ["MANIFEST.json is not valid JSON"]}
        if manifest.get("format") != FORMAT:
            problems.append(f"unknown format {manifest.get('format')!r}")
        sig = z.read("SIGNATURE").decode(errors="replace").strip()
        if not key:
            problems.append("no site secret available: signature not checked")
        elif not hmac.compare_digest(sig, sign(manifest, key)):
            problems.append("signature does not match the manifest (forged, edited, or another site's key)")
        listed = {f["name"]: f for f in manifest.get("files", [])}
        for n, f in listed.items():
            if n not in names:
                problems.append(f"{n}: listed but missing")
                continue
            info = z.getinfo(n)
            if info.file_size > MAX_BUNDLE:
                problems.append(f"{n}: too large")
                continue
            b = z.read(n)
            if hashlib.sha256(b).hexdigest() != f.get("sha256") or len(b) != f.get("bytes"):
                problems.append(f"{n}: content changed (SHA-256 mismatch)")
        for n in sorted(names - set(listed) - {"MANIFEST.json", "SIGNATURE"}):
            problems.append(f"{n}: not in the manifest (added later)")
    return {"ok": not problems, "problems": problems, "alert_id": manifest.get("alert_id"),
            "exported_by": manifest.get("exported_by"), "exported_utc": manifest.get("exported_utc"),
            "files": sorted(listed)}
