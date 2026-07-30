"""Canonical waste taxonomy and per-source class mappings.

Decision (superseding the 2026-07-18 "merge both" taxonomy recorded in
PLAN.md's history): the project targets the **7-class recycling taxonomy**
used by `Waste-Classification-Tech-Stack-Recommendation.md` /
`Project-Action-Plan.md` — paper, cardboard, plastic, glass, metal, organic,
other — sourced entirely from public datasets. The custom 18-class
medical/PPE-skewed dataset (battery, mask, glove, syringe, iv_bag, cotton,
...) is out of scope for this taxonomy; it isn't referenced below.

This module is the single source of truth for the taxonomy. Every data source
declares how its raw folder names map onto the canonical classes, so
consolidation is a lookup, never an ad-hoc rename scattered through scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WasteClass:
    """A canonical target class."""

    name: str  # canonical id, lowercase snake_case
    display: str  # human-readable label
    note: str = ""


# ---------------------------------------------------------------------------
# Canonical 7-class taxonomy (docs' "paper, cardboard, plastic, glass, metal,
# organic, other/reject" list).
# ---------------------------------------------------------------------------
CLASSES: tuple[WasteClass, ...] = (
    WasteClass("cardboard", "Cardboard"),
    WasteClass("paper", "Paper"),
    WasteClass("plastic", "Plastic"),
    WasteClass("glass", "Glass"),
    WasteClass("metal", "Metal"),
    WasteClass("organic", "Organic"),
    WasteClass("other", "Other / Non-recyclable",
               note="catch-all: general trash, e-waste, textiles, medical, "
                    "battery, and anything else outside the six core "
                    "material classes"),
)

# Ordered canonical class names — this order defines the classifier head indices.
CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in CLASSES)
NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASSES)


def class_index(name: str) -> int:
    try:
        return NAME_TO_INDEX[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"'{name}' is not a canonical class. Known: {CLASS_NAMES}"
        ) from exc


# ---------------------------------------------------------------------------
# Data sources and their raw -> canonical folder mappings.
#
# A mapping value of None means "drop this raw class" (not part of our target).
# Any raw folder not present in a source's `mapping` is treated as an error at
# ingest time, so new/renamed source folders can't be silently mis-binned.
#
# All three sources below are classification (folder-per-class) datasets —
# compatible with `edgewaste.data.ingest`'s ImageFolder-style consolidation.
# TACO and ZeroWaste-f (the docs' detection/localization datasets) are NOT
# folder-per-class and are handled separately by `edgewaste.detect`, not here.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataSource:
    key: str  # short id used on the CLI and in provenance records
    title: str  # human description
    kind: str  # "kaggle" (only kind currently supported by the fetch/ingest pipeline)
    locator: str = ""  # kaggle "owner/dataset-slug"
    # raw class folder (as it appears in the source) -> canonical class or None
    mapping: dict[str, str | None] = field(default_factory=dict)
    note: str = ""


TRASHNET = DataSource(
    key="trashnet",
    title="TrashNet (Kaggle mirror: asdasdasasdas/garbage-classification)",
    kind="kaggle",
    locator="asdasdasasdas/garbage-classification",
    mapping={
        "cardboard": "cardboard",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "trash": "other",
    },
    note="Classic 6-class TrashNet mirror, ~2,527 images, clean lab "
         "background. Folder nesting varies by release (commonly "
         "'Garbage classification/Garbage classification/<class>/') — ingest "
         "walks the whole tree matching on leaf folder name, so nesting "
         "depth doesn't matter.",
)

TRASHBOX = DataSource(
    key="trashbox",
    title="TrashBox (Kaggle mirror: minhle13/trashbox)",
    kind="kaggle",
    locator="minhle13/trashbox",
    mapping={
        "cardboard": "cardboard",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "e-waste": "other",
        "medical": "other",
    },
    note="~14.3k in-the-wild/web-sourced images across 7 classes; fills "
         "TrashNet's diversity gap per the tech-stack doc's combination "
         "strategy. Verify folder names after download (e-waste/medical "
         "spelling can vary by mirror release).",
)

GARBAGE_12 = DataSource(
    key="garbage12",
    title="Garbage Classification 12 classes (Kaggle: mostafaabla/garbage-classification)",
    kind="kaggle",
    locator="mostafaabla/garbage-classification",
    mapping={
        "paper": "paper",
        "cardboard": "cardboard",
        "biological": "organic",
        "metal": "metal",
        "plastic": "plastic",
        "green-glass": "glass",
        "brown-glass": "glass",
        "white-glass": "glass",
        "clothes": "other",
        "shoes": "other",
        "battery": "other",
        "trash": "other",
    },
    note="Primary source for 'organic' (its 'biological' folder) — neither "
         "TrashNet nor TrashBox has an organic/food-waste class.",
)

SOURCES: dict[str, DataSource] = {
    s.key: s for s in (TRASHNET, TRASHBOX, GARBAGE_12)
}

# ---------------------------------------------------------------------------
# Known gap, surfaced deliberately rather than hidden: the tech-stack doc also
# names ZeroWaste-f and WaRP-C as useful additions (closest domain match to an
# actual conveyor deployment). Neither has a confirmed Kaggle mirror as of
# 2026-07 — ZeroWaste-f is distributed from http://ai.bu.edu/zerowaste/ under
# manual download, not a simple `kaggle datasets download` call — so they are
# NOT wired into SOURCES yet. Add them by hand (download_dir + a `local`-kind
# path) if/when you fetch them manually; the ingest pipeline's per-source
# mapping design supports that without other changes.
# ---------------------------------------------------------------------------
ZEROWASTE_F_GAP = (
    "ZeroWaste-f / WaRP-C are not wired into SOURCES: no confirmed Kaggle "
    "mirror as of 2026-07, manual download required from the official "
    "project pages. Add manually if pursued."
)


def summary() -> str:
    """Human-readable taxonomy summary (used by the CLI --show flag)."""
    lines = [f"Canonical taxonomy: {NUM_CLASSES} classes", ""]
    for c in CLASSES:
        lines.append(f"  {NAME_TO_INDEX[c.name]:2d} {c.name:<10} {c.display}"
                      f"{'  - ' + c.note if c.note else ''}")
    lines.append("")
    lines.append("Sources:")
    for s in SOURCES.values():
        mapped = sorted({v for v in s.mapping.values() if v})
        lines.append(f"  {s.key:<10} {s.kind:<7} -> {', '.join(mapped)}")
    lines.append("")
    lines.append("NOTE: " + ZEROWASTE_F_GAP)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(summary())
