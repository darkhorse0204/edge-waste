"""Fine-tune YOLO26 for single-class item localization.

Thin wrapper around the Ultralytics training API — see
docs.ultralytics.com/models/yolo26 for the underlying options. Weights
(``yolo26n.pt``) auto-download from the Ultralytics release on first use.

Usage:
    edgewaste-detect-train --config configs/detect.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Config, DetectConfig


def train(cfg: DetectConfig) -> Path:
    from ultralytics import YOLO

    data_yaml = Path(cfg.data_yaml)
    if not data_yaml.exists():
        raise SystemExit(
            f"{data_yaml} not found. Run `edgewaste-detect-fetch` first to "
            f"download and prepare the TACO dataset."
        )

    model = YOLO(cfg.model)
    results = model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.image_size,
        batch=cfg.batch_size,
        project=cfg.output_dir,
        name="taco_single_class",
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest detector checkpoint: {best}")
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the YOLO26 item-localization detector.")
    ap.add_argument("--config", default="configs/detect.yaml")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config).detect
    train(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
