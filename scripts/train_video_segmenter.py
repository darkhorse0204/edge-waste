# train_video_segmenter.py - fine-tunes a small yolo segmentation model on the zerowaste conveyor-video frames (4 material classes)
"""usage: python scripts/train_video_segmenter.py [--epochs 30] [--imgsz 640] [--batch 8]
needs data/video/zerowaste (made by scripts/prepare_zerowaste_video.py). the best weights are written to
runs/video_seg/zerowaste_seg/weights/best.pt. keep workers at 2 or less: this laptop has a tight memory limit."""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(ROOT / "weights" / "yolo26n-seg.pt"), help="pretrained segmentation weights to start from")
    ap.add_argument("--data", default=str(ROOT / "data/video/zerowaste/data.yaml"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="zerowaste_seg")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.resume:
        YOLO(str(ROOT / "runs/video_seg" / a.name / "weights/last.pt")).train(resume=True)
        return
    model = YOLO(a.base)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=0, project=str(ROOT / "runs/video_seg"),
                name=a.name, patience=8, seed=42, cache=False, amp=True, plots=True,
                flipud=0.5, fliplr=0.5, degrees=10, mosaic=1.0)      # conveyor items can lie in any direction, so flips and turns are fair


if __name__ == "__main__":
    main()
