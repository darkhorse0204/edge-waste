"""Download the public datasets used to fill blueprint-only classes.

These fill the classes the custom dataset lacks (Organic via 'biological',
E-Waste, General Waste via 'trash', plus Shoes). Requires the `kaggle` package
and API credentials in ``~/.kaggle/kaggle.json`` (or the KAGGLE_USERNAME /
KAGGLE_KEY env vars). If neither is available the command explains how to set
them up and exits without error, so the rest of the pipeline still runs on the
custom dataset alone.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..config import Config
from ..taxonomy import SOURCES, DataSource

CREDS_HELP = """\
Kaggle credentials not found. To enable public-dataset download:
  1. Create an API token at https://www.kaggle.com/settings  ("Create New Token")
  2. Save the downloaded kaggle.json to:
        {home}/.kaggle/kaggle.json
     (or set KAGGLE_USERNAME and KAGGLE_KEY environment variables)
Then re-run: edgewaste-fetch
Until then, Stage 1 runs on the custom dataset only (14 of 18 classes).
"""


def _have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    return cfg.exists()


def _kaggle_available() -> bool:
    try:
        import kaggle  # noqa: F401
        return True
    except Exception:
        return False


def download_source(source: DataSource, downloads_dir: Path) -> bool:
    """Download and unzip one kaggle source. Returns True on success."""
    if source.kind != "kaggle":
        return False
    dest = downloads_dir / source.key
    dest.mkdir(parents=True, exist_ok=True)
    # Skip if already populated.
    if any(dest.iterdir()):
        print(f"  [have] {source.key}: already present at {dest}")
        return True
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"  [get ] {source.key}: {source.locator} -> {dest}")
    api.dataset_download_files(source.locator, path=str(dest), unzip=True, quiet=False)
    return True


def fetch_all(cfg: Config) -> int:
    downloads_dir = Path(cfg.data.downloads_dir)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    kaggle_sources = [s for s in SOURCES.values() if s.kind == "kaggle"]
    if not kaggle_sources:
        print("No kaggle sources declared.")
        return 0

    if not _kaggle_available():
        print("The 'kaggle' package is not installed. Install with:\n"
              "  pip install kaggle")
        return 0
    if not _have_credentials():
        print(CREDS_HELP.format(home=Path.home()))
        return 0

    ok = 0
    for source in kaggle_sources:
        try:
            if download_source(source, downloads_dir):
                ok += 1
        except Exception as exc:  # pragma: no cover - network/credential errors
            print(f"  [fail] {source.key}: {exc}")
    print(f"\nDownloaded {ok}/{len(kaggle_sources)} public sources into {downloads_dir}")
    print("Next: run `edgewaste-ingest` to merge them into canonical classes.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download public datasets for the class merge.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    return fetch_all(cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
