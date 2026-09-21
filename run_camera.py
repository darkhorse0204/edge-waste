"""One-command live-camera launcher for the full pipeline.

Pre-flight-checks that the detector and classifier checkpoints exist (exits
with a clear message naming whichever is missing instead of a deep
traceback), stamps a timestamped session directory under runs/logs/, and
runs edgewaste.pipeline in camera mode logging every detection to it.

Usage:
    python run_camera.py
    python run_camera.py --det-ckpt runs/detect/taco_single_class/weights/best.pt \
        --cls-ckpt runs/stage1/best.pt
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from edgewaste.config import Config  # noqa: E402
from edgewaste.pipeline import run_camera  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Live-camera session launcher.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--det-ckpt", default="runs/detect/taco_single_class/weights/best.pt")
    ap.add_argument("--cls-ckpt", default="runs/stage1/best.pt")
    ap.add_argument("--cam-index", type=int, default=0)
    ap.add_argument("--logs-root", default="runs/logs")
    args = ap.parse_args()

    missing = [p for p in (args.config, args.det_ckpt, args.cls_ckpt) if not Path(p).exists()]
    if missing:
        print("Cannot start — missing required file(s):")
        for m in missing:
            print(f"  - {m}")
        print("\nTrain the classifier/detector first (see README.md), or point "
              "--det-ckpt/--cls-ckpt at existing checkpoints.")
        return 1

    session_dir = Path(args.logs_root) / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session directory: {session_dir}")

    cfg = Config.load(args.config)
    run_camera(cfg, args.det_ckpt, args.cls_ckpt, args.cam_index, log_dir=str(session_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
