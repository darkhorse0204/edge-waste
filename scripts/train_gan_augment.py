"""Train a per-class DCGAN and generate synthetic augmentation images.

Usage:
    python scripts/train_gan_augment.py --class-dir path/to/class --class-name textile \
        --epochs 100 --n-samples 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edgewaste.augment_gan import generate_synthetic_images, train_gan
from edgewaste.utils import pick_device


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train a DCGAN for one minority class.")
    ap.add_argument("--class-dir", required=True, help="Folder of real images for this class.")
    ap.add_argument("--class-name", required=True)
    ap.add_argument("--output-dir", default=None,
                     help="Defaults to runs/gan/<class-name>.")
    ap.add_argument("--image-size", type=int, default=64, choices=[32, 64])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=0,
                     help="Synthetic images to generate after training (0 = skip).")
    ap.add_argument("--synthetic-out", default=None,
                     help="Defaults to <class-dir>/../<class-name>_synthetic/")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is free.")
    args = ap.parse_args(argv)

    out_dir = args.output_dir or f"runs/gan/{args.class_name}"
    device = "cpu" if args.cpu else str(pick_device())

    history = train_gan(args.class_dir, out_dir, image_size=args.image_size,
                         epochs=args.epochs, batch_size=args.batch_size, device=device)

    print(f"\nFinal g_loss={history['g_loss'][-1]:.3f}  d_loss={history['d_loss'][-1]:.3f}")
    print(f"Sample grid: {out_dir}/sample_grid.png")

    if args.n_samples > 0:
        synth_out = args.synthetic_out or str(Path(args.class_dir).parent / f"{args.class_name}_synthetic")
        paths = generate_synthetic_images(
            Path(out_dir) / "generator.pt", args.n_samples, synth_out,
            args.class_name, device=device)
        print(f"Generated {len(paths)} synthetic images -> {synth_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
