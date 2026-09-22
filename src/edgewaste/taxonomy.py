"""Canonical waste taxonomy and per-source class mappings.

**Decision (2026-09-23, supersedes the 11-class and earlier 7-class
taxonomies):** the project targets a **33-class fine-grained item taxonomy**
organised into **9 material families**.

Why fine-grained, and why a hierarchy. A flat "plastic / glass / metal"
label is not what a sorting facility acts on: a PET drinks bottle, a
detergent bottle and a plastic bag are all "plastic" but go to different
processes, while a polystyrene cup is not processed with either. So the
classifier predicts the *item* (33 classes) and the family is derived from
it (9 families). Both levels are reported: fine-grained accuracy measures
item recognition, family accuracy measures whether the item would be
physically routed correctly. Confusing `plastic_water_bottles` with
`plastic_soda_bottles` is a fine-grained error but a routing success, and
splitting the metric this way makes that visible instead of hiding it.

The hazardous families exist for a different reason than the rest: battery,
e-waste and medical items must never reach a recycling or compost stream at
all (fire, injury, regulated disposal), so they are separated at the
taxonomy level rather than left inside a catch-all.

This module is the single source of truth for the taxonomy. Every data
source declares how its raw folder names map onto the canonical classes, so
consolidation is a lookup, never an ad-hoc rename scattered through scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WasteClass:
    """A canonical target class."""

    name: str  # canonical id, lowercase snake_case
    display: str  # human-readable label
    family: str  # material family this item rolls up into
    note: str = ""


# ---------------------------------------------------------------------------
# Material families. These are the physical streams a sorter routes to; the
# 33 item classes below each belong to exactly one.
# ---------------------------------------------------------------------------
FAMILIES: tuple[str, ...] = (
    "plastic", "paper", "cardboard", "glass", "metal",
    "organic", "styrofoam", "textile", "hazardous",
)

# Families that must never be routed into a recycling or compost stream.
HAZARDOUS_FAMILIES: frozenset[str] = frozenset({"hazardous"})


# ---------------------------------------------------------------------------
# The 33-class fine-grained taxonomy. Tuple order defines the classifier
# head's index order and is embedded into every checkpoint.
# ---------------------------------------------------------------------------
CLASSES: tuple[WasteClass, ...] = (
    # --- plastic (9) ---
    WasteClass("plastic_water_bottles", "Plastic water bottle", "plastic"),
    WasteClass("plastic_soda_bottles", "Plastic soda bottle", "plastic"),
    WasteClass("plastic_detergent_bottles", "Plastic detergent bottle", "plastic"),
    WasteClass("plastic_food_containers", "Plastic food container", "plastic"),
    WasteClass("plastic_shopping_bags", "Plastic shopping bag", "plastic"),
    WasteClass("plastic_trash_bags", "Plastic trash bag", "plastic"),
    WasteClass("plastic_cup_lids", "Plastic cup lid", "plastic"),
    WasteClass("plastic_straws", "Plastic straw", "plastic"),
    WasteClass("disposable_plastic_cutlery", "Disposable plastic cutlery", "plastic"),
    # --- paper (4) ---
    WasteClass("newspaper", "Newspaper", "paper"),
    WasteClass("magazines", "Magazine", "paper"),
    WasteClass("office_paper", "Office paper", "paper"),
    WasteClass("paper_cups", "Paper cup", "paper",
               note="usually poly-coated, so a family-level 'paper' routing "
                    "is an approximation worth flagging in the report"),
    # --- cardboard (2) — a separate stream from mixed paper in practice ---
    WasteClass("cardboard_boxes", "Cardboard box", "cardboard"),
    WasteClass("cardboard_packaging", "Cardboard packaging", "cardboard"),
    # --- glass (3) ---
    WasteClass("glass_beverage_bottles", "Glass beverage bottle", "glass"),
    WasteClass("glass_food_jars", "Glass food jar", "glass"),
    WasteClass("glass_cosmetic_containers", "Glass cosmetic container", "glass"),
    # --- metal (4) ---
    WasteClass("aluminum_soda_cans", "Aluminium soda can", "metal"),
    WasteClass("aluminum_food_cans", "Aluminium food can", "metal"),
    WasteClass("steel_food_cans", "Steel food can", "metal"),
    WasteClass("aerosol_cans", "Aerosol can", "metal",
               note="pressurised: a puncture hazard upstream of shredding, "
                    "even though the material itself is recyclable metal"),
    # --- organic (4) ---
    WasteClass("food_waste", "Food waste", "organic"),
    WasteClass("eggshells", "Eggshells", "organic"),
    WasteClass("coffee_grounds", "Coffee grounds", "organic"),
    WasteClass("tea_bags", "Tea bags", "organic"),
    # --- styrofoam (2) — EPS is not processed with other plastics ---
    WasteClass("styrofoam_cups", "Styrofoam cup", "styrofoam"),
    WasteClass("styrofoam_food_containers", "Styrofoam food container", "styrofoam"),
    # --- textile (2) ---
    WasteClass("clothing", "Clothing", "textile"),
    WasteClass("shoes", "Shoes", "textile"),
    # --- hazardous (3) — never routed to recycling or compost ---
    WasteClass("battery", "Battery", "hazardous",
               note="fire risk in collection vehicles and sorting plants"),
    WasteClass("e_waste", "E-waste", "hazardous",
               note="WEEE: heavy metals and recoverable rare earths"),
    WasteClass("medical", "Medical waste", "hazardous",
               note="sharps/biohazard risk; incineration or specialised "
                    "treatment"),
)

# Ordered canonical class names — this order defines the classifier head indices.
CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in CLASSES)
NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASSES)

# Item class -> material family, and the reverse index.
CLASS_TO_FAMILY: dict[str, str] = {c.name: c.family for c in CLASSES}
FAMILY_TO_CLASSES: dict[str, tuple[str, ...]] = {
    fam: tuple(c.name for c in CLASSES if c.family == fam) for fam in FAMILIES
}
FAMILY_TO_INDEX: dict[str, int] = {fam: i for i, fam in enumerate(FAMILIES)}
NUM_FAMILIES: int = len(FAMILIES)


def class_index(name: str) -> int:
    try:
        return NAME_TO_INDEX[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"'{name}' is not a canonical class. Known: {CLASS_NAMES}"
        ) from exc


def family_of(name: str) -> str:
    """Material family for an item class."""
    try:
        return CLASS_TO_FAMILY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"'{name}' is not a canonical class. Known: {CLASS_NAMES}"
        ) from exc


def family_index_of(name: str) -> int:
    """Family index for an item class — used for hierarchical evaluation."""
    return FAMILY_TO_INDEX[family_of(name)]


def is_hazardous(name: str) -> bool:
    """Whether this item class belongs to a hazardous family."""
    return CLASS_TO_FAMILY.get(name) in HAZARDOUS_FAMILIES


# ---------------------------------------------------------------------------
# Data sources and their raw -> canonical folder mappings.
#
# A mapping value of None means "drop this raw class". Any raw folder not
# present in a source's `mapping` is treated as an error at ingest time, so
# new/renamed source folders can't be silently mis-binned.
#
# Note on the coarse legacy sources: TrashNet's and TrashBox's plain
# 'plastic'/'glass'/'metal'/'paper' folders are deliberately dropped. A
# folder labelled only 'glass' cannot be assigned to glass_beverage_bottles
# vs glass_food_jars vs glass_cosmetic_containers without inventing a label,
# and mixing a coarse 'glass' class alongside the three fine ones would make
# the class depend on which dataset an image came from rather than on the
# object. Those sources are kept only for the classes they *can* resolve
# unambiguously — which for TrashBox is e-waste and medical, the two
# hazardous streams nothing else covers.
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


RECYCLABLE_HOUSEHOLD = DataSource(
    key="recyclable_household",
    title="Recyclable and Household Waste Classification "
          "(Kaggle: alistairking/recyclable-and-household-waste-classification)",
    kind="kaggle",
    locator="alistairking/recyclable-and-household-waste-classification",
    # Folder names already match the canonical class names one-for-one.
    mapping={name: name for name in (
        "plastic_water_bottles", "plastic_soda_bottles",
        "plastic_detergent_bottles", "plastic_food_containers",
        "plastic_shopping_bags", "plastic_trash_bags", "plastic_cup_lids",
        "plastic_straws", "disposable_plastic_cutlery",
        "newspaper", "magazines", "office_paper", "paper_cups",
        "cardboard_boxes", "cardboard_packaging",
        "glass_beverage_bottles", "glass_food_jars",
        "glass_cosmetic_containers",
        "aluminum_soda_cans", "aluminum_food_cans", "steel_food_cans",
        "aerosol_cans",
        "food_waste", "eggshells", "coffee_grounds", "tea_bags",
        "styrofoam_cups", "styrofoam_food_containers",
        "clothing", "shoes",
    )},
    note="30 balanced item classes (~500 images each, 15k total), the "
         "backbone of the taxonomy. Each class folder holds a 'default' "
         "(studio) and a 'real_world' subfolder; ingest pulls both, and the "
         "split is recoverable from provenance.csv's source_path for a "
         "studio-vs-real-world domain-gap ablation.",
)

TRASHBOX = DataSource(
    key="trashbox",
    title="TrashBox (Kaggle mirror: minhle13/trashbox)",
    kind="kaggle",
    locator="minhle13/trashbox",
    mapping={
        # Coarse material folders: dropped, see the note above.
        "cardboard": None,
        "glass": None,
        "metal": None,
        "paper": None,
        "plastic": None,
        # The two hazardous streams no other source covers.
        "e-waste": "e_waste",
        "medical": "medical",
    },
    note="Kept solely for its e-waste and medical folders — the hazardous "
         "classes the household dataset lacks entirely.",
)

GARBAGE_12 = DataSource(
    key="garbage12",
    title="Garbage Classification 12 classes (Kaggle: mostafaabla/garbage-classification)",
    kind="kaggle",
    locator="mostafaabla/garbage-classification",
    mapping={
        # Coarse material folders: dropped, see the note above.
        "paper": None,
        "cardboard": None,
        "metal": None,
        "plastic": None,
        "green-glass": None,
        "brown-glass": None,
        "white-glass": None,
        "trash": None,
        # Resolvable classes.
        "battery": "battery",
        "clothes": "clothing",
        "shoes": "shoes",
        # 'biological' is food/garden scraps, i.e. this taxonomy's food_waste.
        "biological": "food_waste",
    },
    note="Contributes the battery hazardous class, plus extra textile and "
         "food-waste images on top of the household dataset's.",
)

SOURCES: dict[str, DataSource] = {
    s.key: s for s in (RECYCLABLE_HOUSEHOLD, TRASHBOX, GARBAGE_12)
}

# ---------------------------------------------------------------------------
# TrashNet (asdasdasasdas/garbage-classification) is no longer wired in: all
# six of its folders are coarse material labels with no unambiguous
# fine-grained target, so under this taxonomy it would contribute nothing.
# It remains the right source if the project ever reverts to a flat
# material-only taxonomy.
#
# WaRP (parohod/warp-waste-recycling-plant-dataset) IS now on Kaggle — the
# long-standing "no confirmed mirror" gap is closed. 28 classes of real
# recycling-plant imagery with overlap, deformation and poor lighting; the
# natural next addition for a robustness/hard-case evaluation, at 845 MB.
# Not wired in yet: its classes are mostly fine-grained plastic-bottle
# variants that would need their own mapping decisions.
# ---------------------------------------------------------------------------
WARP_NOTE = (
    "WaRP is available at parohod/warp-waste-recycling-plant-dataset (845 MB, "
    "28 classes, real plant conditions). Candidate for a robustness "
    "evaluation; needs a mapping decision for its plastic-bottle variants."
)


def summary() -> str:
    """Human-readable taxonomy summary (used by the CLI --show flag)."""
    lines = [f"Canonical taxonomy: {NUM_CLASSES} item classes "
             f"in {NUM_FAMILIES} material families", ""]
    for fam in FAMILIES:
        members = FAMILY_TO_CLASSES[fam]
        flag = "  [HAZARDOUS]" if fam in HAZARDOUS_FAMILIES else ""
        lines.append(f"  {fam.upper()}{flag}  ({len(members)})")
        for name in members:
            lines.append(f"    {NAME_TO_INDEX[name]:2d} {name}")
        lines.append("")
    lines.append("Sources:")
    for s in SOURCES.values():
        mapped = sorted({v for v in s.mapping.values() if v})
        dropped = sum(1 for v in s.mapping.values() if v is None)
        lines.append(f"  {s.key:<22} {len(mapped):>2} classes"
                     f"{f' ({dropped} coarse folders dropped)' if dropped else ''}")
    lines.append("")
    lines.append("NOTE: " + WARP_NOTE)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(summary())
