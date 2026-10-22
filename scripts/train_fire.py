"""Fine-tune the fire/smoke detector on CPU (Ultralytics).

Starts from a YOLO11n checkpoint pre-trained on D-Fire (sayedgamal99, MIT;
class order 0=smoke, 1=fire) and adapts it to indoor campus-like scenes
(Home-fire) + public spaces (DBA-Fire) + hard negatives (DeepQuest Neutral).
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="/home/claude/models/fire_smoke/sayedgamal_yolo11/yolo11-d-fire-dataset.pt")
ap.add_argument("--data", default="data/fire_smoke/data.yaml")
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--fraction", type=float, default=1.0)
args = ap.parse_args()
model = YOLO(args.weights)
model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device="cpu", workers=2,
            fraction=args.fraction, freeze=10, cos_lr=True, lr0=0.004, warmup_epochs=0.5, mosaic=1.0,
            close_mosaic=1, patience=10, project="runs/fire", name="ft", exist_ok=True, plots=False, val=True,
            amp=False, verbose=False)
