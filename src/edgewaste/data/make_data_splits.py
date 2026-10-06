# make_data_splits.py - writes the seeded train/val/test split manifest (data/splits.csv)
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

from edgewaste.config import Config
from edgewaste.taxonomy import CLASS_NAMES, class_index

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect(processed_dir: Path) -> dict[str, list[Path]]:
    # provenance.csv is rewritten by every ingest run, so it lists exactly the
    # images that run produced. images from an older run (same photo under an
    # older file name) sit in the folder but are not listed; splitting them
    # too would put the same photo into train and test.
    prov = processed_dir / "provenance.csv"
    current: set[tuple[str, str]] | None = None
    if prov.exists():
        with prov.open(encoding="utf-8") as fh:
            current = {(r["canonical_class"], r["dest_name"]) for r in csv.DictReader(fh)}

    by_class: dict[str, list[Path]] = {}
    stale = 0
    for cls in CLASS_NAMES:
        d = processed_dir / cls
        if not d.exists():
            continue
        files = []
        for p in sorted(d.iterdir()):
            if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS):
                continue
            if current is not None and (cls, p.name) not in current:
                stale += 1
                continue
            files.append(p)
        if files:
            by_class[cls] = files
    if stale:
        print(f"[warn] ignored {stale} stale image(s) in {processed_dir} not produced by "
              f"the latest ingest (older file names of the same photos)")
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
            # too few to stratify into 3 splits — put all in train.
            for f in files:
                rows.append((str(f), idx, cls, "train"))
                split_counts["train"] += 1
            continue
        # first carve out test, then val from the remainder.
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
    ap.add_argument("--config", default="configs/classifier_convnext_vit.yaml")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    build_splits(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
