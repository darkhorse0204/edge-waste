"""Build a stratified train/val/test manifest from the processed dataset.

Scans ``processed_dir/<canonical_class>/*`` and writes a single CSV
(``data/splits.csv``) with columns: ``path,label,class_name,split``. Splitting
is stratified per class so rarer classes (e.g. 'other', which pools several
minor source categories) keep representation in every split, deterministic
given the configured seed.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from ..config import Config
from ..taxonomy import CLASS_NAMES, class_index

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect(processed_dir: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = {}
    for cls in CLASS_NAMES:
        d = processed_dir / cls
        if not d.exists():
            continue
        files = [p for p in sorted(d.iterdir())
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if files:
            by_class[cls] = files
    return by_class


def build_splits(cfg: Config) -> dict[str, int]:
    processed_dir = Path(cfg.data.processed_dir)
    by_class = _collect(processed_dir)
    if not by_class:
        raise SystemExit(
            f"No images under {processed_dir}. Run `edgewaste-ingest` first."
        )

    val_f = cfg.data.val_fraction
    test_f = cfg.data.test_fraction
    seed = cfg.data.seed
    rows: list[tuple[str, int, str, str]] = []
    split_counts: dict[str, int] = defaultdict(int)

    for cls, files in by_class.items():
        idx = class_index(cls)
        n = len(files)
        if n < 3:
            # Too few to stratify into 3 splits — put all in train.
            for f in files:
                rows.append((str(f), idx, cls, "train"))
                split_counts["train"] += 1
            continue
        # First carve out test, then val from the remainder.
        train_val, test = train_test_split(
            files, test_size=test_f, random_state=seed, shuffle=True)
        rel_val = val_f / (1.0 - test_f)
        train, val = train_test_split(
            train_val, test_size=rel_val, random_state=seed, shuffle=True)
        for f in train:
            rows.append((str(f), idx, cls, "train")); split_counts["train"] += 1
        for f in val:
            rows.append((str(f), idx, cls, "val")); split_counts["val"] += 1
        for f in test:
            rows.append((str(f), idx, cls, "test")); split_counts["test"] += 1

    manifest = Path(cfg.data.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "class_name", "split"])
        w.writerows(rows)

    print(f"Manifest written to {manifest}")
    print(f"  train={split_counts['train']}  val={split_counts['val']}  "
          f"test={split_counts['test']}  ({len(by_class)} classes present)")
    return dict(split_counts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build stratified train/val/test manifest.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    build_splits(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
