#!/bin/sh
# Continuation run (v2 -> 15 epochs total); restart-safe: re-running resumes from last.pt.
cd "$(dirname "$0")/.." || exit 1
if ls runs/detect/runs/fire/ft3/weights/last.pt >/dev/null 2>&1; then
  exec python3 scripts/train_fire.py --name ft3 --resume
fi
exec python3 scripts/train_fire.py --weights models/fire_smoke.pt --epochs 10 --lr0 0.002 --name ft3
