"""Canonical waste taxonomy and per-source class mappings.

Background (see PLAN.md "Open Questions" and docs/stage-1-core-mvp.md):
the custom on-disk dataset uses a 14-class medical/PPE-skewed taxonomy, while
the blueprint's Module 12 names a general-recycling taxonomy (Organic, Plastic,
Paper, Glass, Metal, Cardboard, Textile, E-Waste, Hazardous, General Waste).
The chosen resolution (2026-07-18) is to **merge both** into one combined
taxonomy: keep every custom class and add the blueprint classes the custom set
lacks, sourcing the additions from public datasets.

This module is the single source of truth for that merged taxonomy. Every data
source declares how its raw folder names map onto the canonical classes below,
so consolidation is a lookup, never an ad-hoc rename scattered through scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WasteClass:
    """A canonical target class."""

    name: str  # canonical id, lowercase snake_case
    display: str  # human-readable label
    group: str  # coarse grouping, useful for reports / decision engine later
    blueprint: bool = False  # named in the blueprint's Module 12 list
    note: str = ""


# ---------------------------------------------------------------------------
# Canonical merged taxonomy (18 classes).
#
# Groups:
#   recyclable  - dry recyclables that also appear in the blueprint's list
#   organic     - biodegradable / food / green waste  (blueprint: "Organic")
#   medical     - PPE / clinical items unique to the custom dataset
#   special     - battery / e-waste / hazardous / general — blueprint additions
# ---------------------------------------------------------------------------
CLASSES: tuple[WasteClass, ...] = (
    # --- recyclables (custom + blueprint overlap) ---
    WasteClass("cardboard", "Cardboard", "recyclable", blueprint=True),
    WasteClass("paper", "Paper", "recyclable", blueprint=True),
    WasteClass("plastic", "Plastic", "recyclable", blueprint=True),
    WasteClass("glass", "Glass", "recyclable", blueprint=True),
    WasteClass("metal", "Metal", "recyclable", blueprint=True),
    WasteClass("textile", "Textile", "recyclable", blueprint=True),
    WasteClass("styrofoam", "Styrofoam", "recyclable",
               note="expanded polystyrene; blueprint folds this under plastic/general"),
    # --- organic (custom 'biodegradable' == blueprint 'Organic') ---
    WasteClass("organic", "Organic", "organic", blueprint=True,
               note="custom dataset's 'biodegradable' folder maps here"),
    # --- medical / PPE (custom only; no blueprint equivalent) ---
    WasteClass("mask", "Face Mask", "medical"),
    WasteClass("glove", "Glove", "medical"),
    WasteClass("syringe", "Syringe", "medical"),
    WasteClass("iv_bag", "IV Bag/Line", "medical",
               note="custom dataset's 'I.V' folder"),
    WasteClass("cotton", "Cotton/Gauze", "medical"),
    # --- special: blueprint additions + battery ---
    WasteClass("battery", "Battery", "special",
               note="present in custom dataset; also a hazardous/e-waste item"),
    WasteClass("e_waste", "E-Waste", "special", blueprint=True,
               note="sourced from public e-waste dataset"),
    WasteClass("hazardous", "Hazardous", "special", blueprint=True,
               note="NO clean public source identified yet - see DATA gap note"),
    WasteClass("general_waste", "General Waste", "special", blueprint=True,
               note="non-recyclable residual; public 'trash' class maps here"),
    WasteClass("shoes", "Shoes/Footwear", "recyclable",
               note="present in public garbage-classification set; kept distinct from textile"),
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
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataSource:
    key: str  # short id used on the CLI and in provenance records
    title: str  # human description
    kind: str  # "local" | "kaggle" | "github"
    # For kaggle sources: the "owner/dataset-slug"; for github: repo URL.
    locator: str = ""
    # raw class folder (as it appears in the source) -> canonical class or None
    mapping: dict[str, str | None] = field(default_factory=dict)
    note: str = ""


# The custom dataset already on disk (14 folders). This is the only source that
# is guaranteed present; the rest require download + credentials.
CUSTOM_DATASET = DataSource(
    key="custom",
    title="Custom Multiclass Waste Dataset (Roboflow-exported, on disk)",
    kind="local",
    locator="",  # path supplied at runtime (see config.custom_dataset_dir)
    mapping={
        "battery": "battery",
        "biodegradable": "organic",
        "cardboard": "cardboard",
        "cotton": "cotton",
        "glass": "glass",
        "glove": "glove",
        "I.V": "iv_bag",
        "mask": "mask",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "styrofoam": "styrofoam",
        "syringe": "syringe",
        "textile": "textile",
    },
    note="8,686 images; class imbalance in textile/I.V/syringe.",
)

# Public datasets used to fill the blueprint classes the custom set lacks
# (Organic, E-Waste, General Waste). Folder names below reflect each dataset's
# published layout; verify after download and adjust if the maintainer changed
# them. Hazardous has no clean single-source public dataset yet (see note).
GARBAGE_12 = DataSource(
    key="garbage12",
    title="Garbage Classification 12 classes (mostafaabla)",
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
        "clothes": "textile",
        "shoes": "shoes",
        "battery": "battery",
        "trash": "general_waste",
    },
    note="Fills Organic (biological), General Waste (trash), Shoes.",
)

EWASTE = DataSource(
    key="ewaste",
    title="E-Waste Image Dataset (akshat103)",
    kind="kaggle",
    locator="akshat103/e-waste-image-dataset",
    mapping={
        # 10 electronic-item subfolders all collapse to e_waste.
        "Battery": "battery",
        "Keyboard": "e_waste",
        "Microwave": "e_waste",
        "Mobile": "e_waste",
        "Mouse": "e_waste",
        "PCB": "e_waste",
        "Player": "e_waste",
        "Printer": "e_waste",
        "Television": "e_waste",
        "Washing Machine": "e_waste",
    },
    note="Folder names vary by release; ingest tolerates case/space differences.",
)

SOURCES: dict[str, DataSource] = {
    s.key: s for s in (CUSTOM_DATASET, GARBAGE_12, EWASTE)
}

# ---------------------------------------------------------------------------
# Known data gap (surfaced deliberately rather than hidden):
#   'hazardous' has no clean, single, freely-downloadable public dataset that
#   maps cleanly. Options for later: (a) treat battery+e_waste as the hazardous
#   umbrella and drop the standalone 'hazardous' class, or (b) hand-curate a
#   small hazardous set (paint cans, chemical bottles, aerosols). Until then the
#   'hazardous' class will simply have zero samples and the trainer will warn.
# ---------------------------------------------------------------------------
HAZARDOUS_GAP = (
    "No public source mapped to 'hazardous' yet; class will be empty until a "
    "source is added or the class is merged into battery/e_waste."
)


def summary() -> str:
    """Human-readable taxonomy summary (used by the CLI --show flag)."""
    lines = [f"Canonical taxonomy: {NUM_CLASSES} classes", ""]
    by_group: dict[str, list[WasteClass]] = {}
    for c in CLASSES:
        by_group.setdefault(c.group, []).append(c)
    for group, members in by_group.items():
        lines.append(f"[{group}]")
        for c in members:
            flag = " (blueprint)" if c.blueprint else ""
            lines.append(f"  {NAME_TO_INDEX[c.name]:2d} {c.name:<14} {c.display}{flag}")
        lines.append("")
    lines.append("Sources:")
    for s in SOURCES.values():
        mapped = sorted({v for v in s.mapping.values() if v})
        lines.append(f"  {s.key:<10} {s.kind:<7} -> {', '.join(mapped)}")
    lines.append("")
    lines.append("NOTE: " + HAZARDOUS_GAP)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(summary())
