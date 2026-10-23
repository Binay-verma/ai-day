#!/bin/sh
# Hard-negative fine-tune: shipped 15-epoch model + mined confounders (scripts/mine_hard_negatives.py),
# 4 epochs at lr0 0.001. Restart-safe: re-running resumes from last.pt.
# Checkpoint rule fixed in advance: last.pt (validation has almost no negatives, so "best" by val mAP
# would not measure false positives).
cd "$(dirname "$0")/.." || exit 1
if ls runs/detect/runs/fire/hn/weights/last.pt >/dev/null 2>&1 && ! grep -q "4/4" logs/train_fire_hn.done 2>/dev/null; then
  exec python3 scripts/train_fire.py --name hn --resume
fi
exec python3 scripts/train_fire.py --weights models/fire_smoke.pt --epochs 4 --lr0 0.001 --name hn
