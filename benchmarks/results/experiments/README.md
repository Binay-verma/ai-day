# Negative results (kept so nobody repeats them)

| Experiment | Protocol | Result | Kept? |
|---|---|---|---|
| Monotone GAM shape functions instead of linear cue weights (`benchmarks/eval_fall_gam.py`, `xai.fusion.fit_gam`) | GMDCSA24 LOSO, YOLO11s-pose, 4 segments, l2 = 3 fixed in advance; sweep reported | F1 0.863 vs 0.886 linear; best of 8 sweep settings 0.872. 5 videos fixed, 8 broken | code kept (exact, tested), not used |
| One-Euro key-point smoothing before fall cues (`keypoint_one_euro.py`) | same | LOSO F1 0.859–0.882 vs 0.886 without | removed |

Conclusion: with the current cues the fall model is limited by pose quality (YOLO11n → 11s gave +0.08),
not by the fusion's functional form or key-point jitter.
