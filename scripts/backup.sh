#!/usr/bin/env bash
# Consistent online backup of the incident database (no images) + calibration.
# Media are encrypted and short-lived by design (retention), so they are not backed up.
# Usage: scripts/backup.sh /mnt/backup   (run daily from cron/systemd timer)
set -euo pipefail
DEST="${1:?destination directory}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DEST"
python3 - "$ROOT/data/edgecopilot.db" "$DEST/edgecopilot-$STAMP.db" <<'PY'
import sqlite3, sys
src = sqlite3.connect(sys.argv[1])
dst = sqlite3.connect(sys.argv[2])
with dst:
    src.backup(dst)          # online, consistent snapshot even while the service writes (WAL)
ok = dst.execute("PRAGMA integrity_check").fetchone()[0]
print("integrity:", ok)
sys.exit(0 if ok == "ok" else 1)
PY
tar -czf "$DEST/calibration-$STAMP.tgz" -C "$ROOT" calibration 2>/dev/null || true
sha256sum "$DEST/edgecopilot-$STAMP.db" > "$DEST/edgecopilot-$STAMP.db.sha256"
# keep 30 days of backups
find "$DEST" -name 'edgecopilot-*.db*' -mtime +30 -delete
find "$DEST" -name 'calibration-*.tgz' -mtime +30 -delete
echo "backup written to $DEST"
