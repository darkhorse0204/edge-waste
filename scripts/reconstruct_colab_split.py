# reconstruct_colab_split.py - rebuilds the exact train/val/test split the 89.30% colab model was trained and tested on
"""Recover the Colab split so the 89.30% checkpoint can be analysed locally
without train images leaking into its test set.

Why this is needed: before the cross-machine hashing fix, each processed
image's file name was md5(str(source path)). On Colab that string was
'data/downloads/<source>/...' with forward slashes; on Windows it has
backslashes, so every name — and therefore the sorted order that the seeded
stratified split consumes — differs, and a locally rebuilt split puts ~70% of
Colab's training images into "test" (the leaky 94.52% re-evaluation).

This script recomputes the Colab-side names from the local provenance file,
optionally re-adds images the local ingest rejected as unreadable (Colab's
processed folder on Drive still held copies from a session before that filter
existed, and the split counted them), and replays the exact split code.

Verified: the 89.30% checkpoint scores exactly 3779/4232 = 89.30% item,
94.14% family and 690/738 hazard recall (fp32) on the rebuilt test split —
all three Colab-reported numbers — only when the one corrupt TrashBox e-waste
image is re-added; without it the e-waste class reshuffles and scores 89.53%.

Usage:
    python scripts/reconstruct_colab_split.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edgewaste.config import Config  # noqa: E402
from edgewaste.data import ingest_raw_datasets as ingest  # noqa: E402
from edgewaste.taxonomy import CLASS_NAMES, SOURCES, class_index  # noqa: E402


def colab_name(source: str, raw_class: str, src_path: str) -> str:
    posix = PureWindowsPath(src_path).as_posix()
    return (f"{source}__{raw_class}__{hashlib.md5(posix.encode()).hexdigest()[:10]}"
            f"{PurePosixPath(posix).suffix.lower()}")


def walk_all_images(cfg: Config) -> list[tuple[str, str, str, str]]:
    """(canonical, source, raw_class, src_path) for every image ingest would
    consider, readable or not — ingest's own walk with the filter and the
    copy step switched off."""
    rows: list[tuple[str, str, str, str]] = []

    class _Collect:
        def writerow(self, r):
            rows.append((r[0], r[1], r[2], r[3]))

    ingest._is_readable_image, ingest._link_or_copy = (lambda p: True), (lambda a, b: None)
    for key, source in SOURCES.items():
        if source.kind != "kaggle":
            continue
        ingest.ingest_source(source, Path(cfg.data.downloads_dir) / key,
                             Path(cfg.data.processed_dir), _Collect())
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/classifier_convnext_vit.yaml")
    ap.add_argument("--exclude-rejected", action="store_true",
                    help="leave out images the local ingest rejected (does NOT match Colab)")
    ap.add_argument("--out", default="data/splits_colab_reconstructed.csv")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)

    prov = list(csv.DictReader(open(Path(cfg.data.processed_dir) / "provenance.csv", encoding="utf-8")))
    accepted = {r["source_path"] for r in prov}
    entries = [(r["canonical_class"], colab_name(r["source"], r["raw_class"], r["source_path"]),
                f"{cfg.data.processed_dir}/{r['canonical_class']}/{r['dest_name']}") for r in prov]
    if not args.exclude_rejected:
        rejected = [r for r in walk_all_images(cfg) if r[3] not in accepted]
        print(f"re-adding {len(rejected)} rejected (unreadable) images:",
              dict(pd.Series([r[0] for r in rejected]).value_counts()))
        entries += [(c, colab_name(s, rc, p), p) for c, s, rc, p in rejected]

    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cls, name, path in entries:
        by_class[cls].append((name, path))
    rows = []
    for cls in CLASS_NAMES:
        items = sorted(by_class.get(cls, []))  # sorted(d.iterdir()) order on Colab
        names = [n for n, _ in items]
        path_of = dict(items)
        trv, te = train_test_split(names, test_size=cfg.data.test_fraction,
                                   random_state=cfg.data.seed, shuffle=True)
        tr, va = train_test_split(trv, test_size=cfg.data.val_fraction / (1 - cfg.data.test_fraction),
                                  random_state=cfg.data.seed, shuffle=True)
        for split, names_ in (("train", tr), ("val", va), ("test", te)):
            rows += [(path_of[n], class_index(cls), cls, split) for n in names_]
    df = pd.DataFrame(rows, columns=["path", "label", "class_name", "split"])
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {df.split.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
