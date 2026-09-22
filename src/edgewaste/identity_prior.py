"""Object-Identity Prior: constrain material predictions by object identity.

The two vision stages predict different things from different evidence — the
detector names the *object* (Can, Bottle, Cigarette), the classifier names
the *material* (metal, glass, plastic). Physics couples them: a Can is not
made of organic matter, a Pop tab is not cardboard. Treating the detector's
output as a prior over materials therefore turns two independent predictions
into a mutual consistency check, for free, with no extra model.

This does two useful things:

1. **Correction.** A classifier that is unsure between metal and organic on a
   crop the detector confidently calls a Can should be pushed toward metal.
   Implausible materials are down-weighted, not forbidden, so a genuinely
   surprising-but-correct reading can still win if the classifier is
   confident enough. Hard-masking would make the system unable to ever report
   a mislabelled object, which is worse than the error it prevents.

2. **Contradiction flagging.** When the two stages still disagree after
   re-weighting, that disagreement is *information*: it usually means the
   crop is bad, the item is genuinely ambiguous, or one stage is wrong.
   Those items are routed to manual review rather than silently sorted.

The prior is a soft compatibility table, not a hard rule set. Weight 1.0 =
expected, IMPLAUSIBLE_WEIGHT = physically odd. Objects whose identity says
nothing about material ('Other litter', 'Unlabeled litter') are uniform, so
they leave the classifier's own distribution untouched.
"""

from __future__ import annotations

import torch

from .taxonomy import CLASS_NAMES, FAMILY_TO_CLASSES, is_hazardous

# How much to down-weight a material the object identity makes implausible.
# Not zero: see the module docstring on hard-masking.
IMPLAUSIBLE_WEIGHT = 0.15

# object class (as the 18-class TACO detector names it) -> plausible materials.
# An empty set means "uninformative": every material stays at weight 1.0.
# Entries may name either a material *family* (expanded to all its item
# classes) or specific item classes, whichever the object identity actually
# pins down. 'Straw' names one item exactly; 'Can' names a family.
OBJECT_MATERIAL_PRIOR: dict[str, set[str]] = {
    "Aluminium foil": {"metal"},
    "Bottle cap": {"plastic_cup_lids", "metal"},
    # A bottle silhouette is genuinely ambiguous between plastic and glass —
    # exactly the confusion the project's own problem statement calls out.
    "Bottle": {"plastic_water_bottles", "plastic_soda_bottles",
               "plastic_detergent_bottles", "glass_beverage_bottles"},
    "Broken glass": {"glass"},
    "Can": {"metal"},
    "Carton": {"cardboard", "paper"},
    "Cigarette": set(),  # no item class covers cigarette butts
    "Cup": {"paper_cups", "styrofoam_cups", "plastic_cup_lids"},
    "Lid": {"plastic_cup_lids", "metal"},
    "Other litter": set(),
    "Other plastic": {"plastic", "styrofoam"},
    "Paper": {"paper"},
    "Plastic bag - wrapper": {"plastic_shopping_bags", "plastic_trash_bags"},
    "Plastic container": {"plastic_food_containers", "plastic_detergent_bottles"},
    "Pop tab": {"metal"},
    "Straw": {"plastic_straws"},
    # Expanded polystyrene: its own family, not processed with other plastics.
    "Styrofoam piece": {"styrofoam"},
    "Unlabeled litter": set(),
}

# The detector is trained on outdoor litter and has no object class for the
# hazardous stream. A battery or syringe therefore arrives as 'Other
# litter'/'Unlabeled litter' — both uninformative, so the prior stays uniform
# and the classifier's hazardous call survives untouched. Belt and braces:
# hazardous classes are additionally exempted below, so the prior can never
# talk the classifier *out of* flagging a hazard.
HAZARD_SAFE_MATERIALS: frozenset[str] = frozenset(
    name for name in CLASS_NAMES if is_hazardous(name)
)


def _expand(entries: set[str]) -> set[str]:
    """Resolve a prior entry set (families and/or item classes) to item classes."""
    resolved: set[str] = set()
    for entry in entries:
        if entry in FAMILY_TO_CLASSES:
            resolved.update(FAMILY_TO_CLASSES[entry])
        else:
            resolved.add(entry)
    return resolved


def prior_vector(object_class: str, class_names: list[str] | tuple[str, ...] = CLASS_NAMES
                  ) -> torch.Tensor:
    """Per-material weights implied by an object class. Uniform if unknown."""
    entries = OBJECT_MATERIAL_PRIOR.get(object_class)
    if not entries:  # unknown object, or an uninformative catch-all
        return torch.ones(len(class_names))
    plausible = _expand(entries)
    return torch.tensor([
        # Hazardous materials are never down-weighted: the prior may not
        # suppress a hazard flag (see HAZARD_SAFE_MATERIALS).
        1.0 if (name in plausible or name in HAZARD_SAFE_MATERIALS)
        else IMPLAUSIBLE_WEIGHT
        for name in class_names
    ])


def apply_identity_prior(
    material_probs: torch.Tensor, object_class: str,
    class_names: list[str] | tuple[str, ...] = CLASS_NAMES,
) -> torch.Tensor:
    """Re-weight a material distribution by the object identity, renormalised."""
    weights = prior_vector(object_class, class_names).to(material_probs.device)
    adjusted = material_probs * weights
    total = adjusted.sum()
    if total <= 0:  # degenerate; fall back to the unmodified distribution
        return material_probs
    return adjusted / total


def is_consistent(object_class: str, material: str) -> bool:
    """Whether this object/material pairing is physically plausible."""
    entries = OBJECT_MATERIAL_PRIOR.get(object_class)
    if not entries:
        return True  # uninformative object identity can't contradict anything
    if material in HAZARD_SAFE_MATERIALS:
        return True  # a hazard call is never treated as a contradiction
    return material in _expand(entries)


def explain(object_class: str, material: str) -> str:
    """One-line human-readable verdict, for logs and the report."""
    entries = OBJECT_MATERIAL_PRIOR.get(object_class)
    if not entries:
        return f"'{object_class}' implies no material constraint"
    if material in HAZARD_SAFE_MATERIALS:
        return (f"hazard call '{material}' kept over object identity "
                f"'{object_class}' - hazard flags are never overridden")
    if material in _expand(entries):
        return f"'{material}' is consistent with '{object_class}'"
    return (f"CONTRADICTION: '{object_class}' implies {sorted(entries)}, "
            f"classifier said '{material}'")
