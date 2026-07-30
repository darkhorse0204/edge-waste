"""Download + prepare the TACO detection dataset for YOLO26 training.

TACO (Trash Annotations in Context) is a litter-detection dataset with ~60
fine-grained categories. This project's detector only needs to answer "is
there an item here, and where" — material typing happens downstream in the
Stage 1 classifier on the cropped detection — so every TACO category is
collapsed to a single class, ``waste_item`` (class index 0), rather than
carrying ~60 categories through the detector.

Source: the pre-converted-to-YOLO-format Kaggle mirror
``vencerlanz09/taco-dataset-yolo-format`` (bounding boxes already as
normalized YOLO .txt files, so no COCO-JSON parsing is needed here). Its exact
internal folder layout isn't something to assume blindly — this module
discovers image/label pairs by matching filename stems anywhere under the
downloaded tree, which is robust to whatever nesting the zip happens to use.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from ..config import Config, DetectConfig

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Sidecar .txt files that are metadata, not per-image YOLO labels.
NON_LABEL_TXT_NAMES = {"classes.txt", "readme.txt", "notes.txt", "data.txt"}


def _have_credentials() -> bool:
    import os
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def download(cfg: DetectConfig) -> bool:
    """Download + unzip the TACO YOLO-format Kaggle mirror. Returns True on success."""
    raw_dir = Path(cfg.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if any(raw_dir.iterdir()):
        print(f"  [have] taco: already present at {raw_dir}")
        return True
    try:
        import kaggle  # noqa: F401
    except Exception:
        print("The 'kaggle' package is not installed. Install with:\n"
              "  pip install kaggle")
        return False
    if not _have_credentials():
        print("Kaggle credentials not found (~/.kaggle/kaggle.json or "
              "KAGGLE_USERNAME/KAGGLE_KEY). See kaggle_sources.py's "
              "CREDS_HELP for setup steps.")
        return False
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"  [get ] taco: {cfg.kaggle_locator} -> {raw_dir}")
    api.dataset_download_files(cfg.kaggle_locator, path=str(raw_dir),
                                unzip=True, quiet=False)
    return True


def _find_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    """Match image files to same-stem YOLO .txt label files anywhere under raw_dir."""
    images_by_stem: dict[str, Path] = {}
    labels_by_stem: dict[str, Path] = {}
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            images_by_stem.setdefault(p.stem, p)
        elif p.suffix.lower() == ".txt" and p.name.lower() not in NON_LABEL_TXT_NAMES:
            labels_by_stem.setdefault(p.stem, p)
    pairs = [(img, labels_by_stem[stem]) for stem, img in images_by_stem.items()
              if stem in labels_by_stem]
    return pairs


def _collapse_and_write_label(src_label: Path, dst_label: Path) -> None:
    """Rewrite a YOLO label file with every class index forced to 0."""
    lines_out = []
    for line in src_label.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue  # malformed/empty line
        _, cx, cy, w, h, *rest = parts
        lines_out.append(" ".join(["0", cx, cy, w, h, *rest]))
    dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))


def prepare(cfg: DetectConfig, val_fraction: float = 0.15, seed: int = 42) -> dict[str, int]:
    """Collapse TACO to single-class labels and lay out the standard Ultralytics tree.

    Writes:
        prepared_dir/images/{train,val}/*
        prepared_dir/labels/{train,val}/*   (single-class YOLO labels)
        prepared_dir/data.yaml
    """
    raw_dir = Path(cfg.raw_dir)
    prepared_dir = Path(cfg.prepared_dir)
    pairs = _find_pairs(raw_dir)
    if not pairs:
        raise SystemExit(
            f"No image/label pairs found under {raw_dir}. Run the download "
            f"step first (edgewaste-detect-fetch), and if it already ran, "
            f"inspect {raw_dir} — the mirror's layout may not match the "
            f"'same-stem image + .txt' assumption this script relies on."
        )
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_fraction))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    counts: dict[str, int] = {}
    for split, split_pairs in splits.items():
        img_dir = prepared_dir / "images" / split
        lbl_dir = prepared_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl in split_pairs:
            dst_img = img_dir / img.name
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            _collapse_and_write_label(lbl, lbl_dir / (img.stem + ".txt"))
        counts[split] = len(split_pairs)

    data_yaml = Path(cfg.data_yaml)
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(
        f"path: {prepared_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['{cfg.class_name}']\n"
    )
    print(f"Prepared {sum(counts.values())} pairs -> {prepared_dir} "
          f"(train={counts['train']}, val={counts['val']})")
    print(f"data.yaml written to {data_yaml}")
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch + prepare the TACO detection dataset.")
    ap.add_argument("--config", default="configs/detect.yaml")
    ap.add_argument("--skip-download", action="store_true",
                     help="Assume raw_dir is already populated.")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config).detect
    if not args.skip_download:
        if not download(cfg):
            return 1
    prepare(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
