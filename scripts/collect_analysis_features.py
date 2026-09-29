# collect_analysis_features.py - caches classifier outputs and features for every split, verifying the test accuracy first
"""Run the one-time feature collection that the ML analysis suite reads.

Usage:
    python scripts/collect_analysis_features.py                       # main 89.30% model
    python scripts/collect_analysis_features.py --expect-acc 0.8930   # fail fast if the split is wrong

The test split is collected first and its accuracy printed (and checked
against --expect-acc), so a wrong manifest is caught in minutes rather than
after the ~1 hour train-split pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edgewaste.analysis.collect_predictions import collect_split, load_split  # noqa: E402
from edgewaste.config import Config  # noqa: E402
from edgewaste.taxonomy import CLASS_TO_FAMILY, is_hazardous  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/classifier_convnext_vit.yaml")
    ap.add_argument("--ckpt", default="runs/stage1/best.pt")
    ap.add_argument("--manifest", default="data/splits_colab_reconstructed.csv")
    ap.add_argument("--out-dir", default="runs/analysis/convnext_vit")
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"])
    ap.add_argument("--expect-acc", type=float, default=None)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    for split in args.splits:
        path = Path(args.out_dir) / f"cache_{split}.npz"
        if path.exists():
            print(f"{split}: cached at {path}, skipping")
        else:
            collect_split(cfg, args.ckpt, args.manifest, split, path)
        d = load_split(path)
        names = d["class_names"]
        y, p = d["labels"], d["probs"].argmax(1)
        fam = np.array([CLASS_TO_FAMILY[n] for n in names])
        haz = np.array([is_hazardous(n) for n in names])
        acc = float((y == p).mean())
        print(f"{split}: n={len(y)} item acc {acc:.4f}  family acc {(fam[y] == fam[p]).mean():.4f}  "
              f"hazard {int(haz[p][haz[y]].sum())}/{int(haz[y].sum())}")
        if split == "test" and args.expect_acc is not None and abs(acc - args.expect_acc) > 5e-5:
            print(f"test accuracy {acc:.4f} != expected {args.expect_acc:.4f}: wrong split manifest, stopping")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
