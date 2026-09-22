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

from .taxonomy import CLASS_NAMES

# How much to down-weight a material the object identity makes implausible.
# Not zero: see the module docstring on hard-masking.
IMPLAUSIBLE_WEIGHT = 0.15

# object class (as the 18-class TACO detector names it) -> plausible materials.
# An empty set means "uninformative": every material stays at weight 1.0.
OBJECT_MATERIAL_PRIOR: dict[str, set[str]] = {
    "Aluminium foil": {"metal"},
    "Bottle cap": {"plastic", "metal"},
    "Bottle": {"plastic", "glass"},
    "Broken glass": {"glass"},
    "Can": {"metal"},
    "Carton": {"cardboard", "paper"},
    "Cigarette": {"other"},
    "Cup": {"paper", "plastic"},
    "Lid": {"plastic", "metal"},
    "Other litter": set(),
    "Other plastic": {"plastic"},
    "Paper": {"paper"},
    "Plastic bag - wrapper": {"plastic"},
    "Plastic container": {"plastic"},
    "Pop tab": {"metal"},
    "Straw": {"plastic"},
    # Styrofoam is expanded polystyrene, i.e. a plastic.
    "Styrofoam piece": {"plastic"},
    "Unlabeled litter": set(),
}


def prior_vector(object_class: str, class_names: list[str] | tuple[str, ...] = CLASS_NAMES
                  ) -> torch.Tensor:
    """Per-material weights implied by an object class. Uniform if unknown."""
    plausible = OBJECT_MATERIAL_PRIOR.get(object_class)
    if not plausible:  # unknown object, or an uninformative catch-all
        return torch.ones(len(class_names))
    return torch.tensor([
        1.0 if name in plausible else IMPLAUSIBLE_WEIGHT for name in class_names
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
    plausible = OBJECT_MATERIAL_PRIOR.get(object_class)
    if not plausible:
        return True  # uninformative object identity can't contradict anything
    return material in plausible


def explain(object_class: str, material: str) -> str:
    """One-line human-readable verdict, for logs and the report."""
    plausible = OBJECT_MATERIAL_PRIOR.get(object_class)
    if not plausible:
        return f"'{object_class}' implies no material constraint"
    if material in plausible:
        return f"'{material}' is consistent with '{object_class}'"
    return (f"CONTRADICTION: '{object_class}' implies "
            f"{sorted(plausible)}, classifier said '{material}'")
