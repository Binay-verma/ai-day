"""Export a test session for review (e.g. after a webcam test).

Writes one zip with:

* ``session.json`` - every alert of the last ``--hours`` (hazard, time, confidence, ALARM/VERIFY,
  the explanation, the evidence ledger, counterfactuals, grounding check, operator decision),
  incident grouping, outcome counts, the audit trail and audit-chain check, and the machine
  (OS, CPU, Python and library versions, model files and sizes);
* ``snapshots/`` - the alerts' privacy-filtered snapshots and evidence heat maps
  (only with ``--with-images``; faces are already pixelated, but check before sharing).

Nothing leaves the computer unless you send the zip yourself.

    python scripts/export_session.py --hours 6 --with-images
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgecopilot.handover import correlate  # noqa: E402
from edgecopilot.security import load_secret  # noqa: E402
from edgecopilot.service import utc_iso  # noqa: E402
from edgecopilot.storage import Store  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def versions() -> dict:
    out = {"python": sys.version.split()[0], "os": platform.platform(), "machine": platform.machine(),
           "cpu_count": os.cpu_count()}
    for mod in ("numpy", "cv2", "onnxruntime", "fastapi", "pydantic"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            out[mod] = None
    models = os.path.join(ROOT, "models")
    out["models"] = {f: os.path.getsize(os.path.join(models, f)) for f in sorted(os.listdir(models))
                     if f.endswith(".onnx")} if os.path.isdir(models) else {}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "edgecopilot.db"))
    ap.add_argument("--media", default=os.path.join(ROOT, "data", "media"))
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--with-images", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "session_export.zip"))
    a = ap.parse_args()
    if not os.path.exists(a.db):
        raise SystemExit(f"No database at {a.db}: run the server at least once first.")
    # the server keys the audit chain with its secret (env var or data/.secret): needed to verify it
    store = Store(a.db, a.media, audit_key=load_secret(path=os.path.join(os.path.dirname(a.db), ".secret")))
    since = time.time() - a.hours * 3600
    alerts = store.list_alerts(5000, since=since)
    keep = ("id", "camera_id", "zone", "hazard", "subject", "t_video", "prob", "peak_prob", "severity", "triage", "status",
            "operator", "operator_note", "explanation", "decision", "verification", "recommendations", "cues", "cleared_at")
    rows = []
    for r in alerts:
        d = {k: r.get(k) for k in keep}
        d["time_utc"] = utc_iso(r.get("wall_time"))
        d["cleared_utc"] = utc_iso(r.get("cleared_at")) if r.get("cleared_at") else None
        rows.append(d)
    counts: dict = {}
    for r in rows:
        counts.setdefault(r["hazard"], {}).setdefault(r["triage"] or "-", 0)
        counts[r["hazard"]][r["triage"] or "-"] += 1
    audit = [x for x in store.audit_log(5000) if (x.get("ts") or 0) >= since]
    session = {"exported_utc": utc_iso(time.time()), "period_hours": a.hours, "n_alerts": len(rows),
               "alerts_by_hazard_and_triage": counts, "incidents": correlate(alerts), "alerts": rows,
               "audit": audit, "audit_chain_check": store.verify_audit_chain(), "machine": versions()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n_img = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps(session, indent=1, default=str))
        if a.with_images:
            for r in alerts:
                for key in ("snapshot_path", "evidence_path"):
                    if r.get(key):
                        data = store.get_media(r[key])
                        if data:
                            z.writestr(f"snapshots/{r['id']}_{key.split('_')[0]}.jpg", data)
                            n_img += 1
    store.close()
    print(f"wrote {a.out}: {len(rows)} alerts, {len(audit)} audit entries, {n_img} images")


if __name__ == "__main__":
    main()
