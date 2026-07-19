"""Consolidate raw data sources into a single canonical-class layout.

Reads each available source (custom on-disk dataset, plus any downloaded public
datasets), maps its raw folders to canonical classes via
``edgewaste.taxonomy``, and writes a flat ``processed_dir/<canonical_class>/``
tree. Files are hard-linked when possible (fast, no extra disk) and copied
otherwise. A ``provenance.csv`` records where every consolidated image came
from, so the merged dataset stays auditable for the report.

The custom source is always ingested. Public sources are ingested only if their
downloaded folder exists under ``downloads_dir`` (see kaggle_sources.py).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path

from ..config import Config
from ..taxonomy import SOURCES, CLASS_NAMES, DataSource

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(folder: Path):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _resolve_raw_class(source: DataSource, raw_name: str) -> str | None:
    """Map a raw folder name to a canonical class, tolerating case/space drift.

    Returns the canonical class name, or None if the source explicitly drops it.
    Raises KeyError if the folder isn't declared in the source mapping at all,
    so an unexpected new folder fails loudly instead of being mis-binned.
    """
    if raw_name in source.mapping:
        return source.mapping[raw_name]
    # tolerant match: normalize case and separators
    norm = raw_name.lower().replace("_", " ").replace("-", " ").strip()
    for key, val in source.mapping.items():
        if key.lower().replace("_", " ").replace("-", " ").strip() == norm:
            return val
    raise KeyError(
        f"Source '{source.key}' has folder '{raw_name}' not in its taxonomy "
        f"mapping. Add it to edgewaste.taxonomy.SOURCES['{source.key}'].mapping "
        f"(map to a canonical class or None to drop)."
    )


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hard link — instant, no extra bytes
    except OSError:
        shutil.copy2(src, dst)


def _dedupe_name(source_key: str, raw_class: str, path: Path) -> str:
    """Stable, collision-resistant output filename that keeps provenance."""
    h = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{source_key}__{raw_class}__{h}{path.suffix.lower()}"


def ingest_source(
    source: DataSource, root: Path, processed_dir: Path, writer: csv.writer
) -> dict[str, int]:
    """Ingest one source rooted at `root`. Returns per-canonical-class counts."""
    counts: dict[str, int] = {}
    if not root.exists():
        print(f"  [skip] {source.key}: {root} not found")
        return counts
    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not subdirs:
        print(f"  [skip] {source.key}: no class subfolders under {root}")
        return counts
    for class_dir in subdirs:
        canonical = _resolve_raw_class(source, class_dir.name)
        if canonical is None:
            continue  # deliberately dropped raw class
        for img in _iter_images(class_dir):
            out_name = _dedupe_name(source.key, class_dir.name, img)
            out_path = processed_dir / canonical / out_name
            _link_or_copy(img, out_path)
            writer.writerow([canonical, source.key, class_dir.name,
                             str(img), out_name])
            counts[canonical] = counts.get(canonical, 0) + 1
    return counts


def ingest(cfg: Config) -> dict[str, int]:
    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = Path(cfg.data.downloads_dir)

    provenance_path = processed_dir / "provenance.csv"
    totals: dict[str, int] = {c: 0 for c in CLASS_NAMES}

    with provenance_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_class", "source", "raw_class",
                         "source_path", "dest_name"])
        for key, source in SOURCES.items():
            if source.kind == "local":
                root = Path(cfg.data.custom_dataset_dir)
            else:
                # public sources unpack to downloads_dir/<key>/
                root = downloads_dir / key
            print(f"- ingesting '{key}' from {root}")
            counts = ingest_source(source, root, processed_dir, writer)
            for cls, n in counts.items():
                totals[cls] += n

    # Report
    print("\nConsolidated per-class counts (processed_dir):")
    grand = 0
    for cls in CLASS_NAMES:
        n = totals[cls]
        grand += n
        flag = "  <-- EMPTY" if n == 0 else ""
        print(f"  {cls:<14} {n:>6}{flag}")
    print(f"  {'TOTAL':<14} {grand:>6}")
    print(f"\nProvenance written to {provenance_path}")
    return totals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Consolidate raw sources into canonical classes.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    ingest(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
