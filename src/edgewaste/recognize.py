"""Object identity as a prior over material class.

The Stage-1 classifier answers "what is this made of" from texture and colour
alone. It has no concept of *what the object is*, so it cannot use the single
most informative fact available: a banana is organic no matter what it looks
like, and a pair of scissors is metal even under bad lighting.

This module adds a COCO-pretrained detector (80 everyday classes, no training)
purely as an identity source, and folds that identity into the material
decision as a *prior*, never as an override:

  certain      banana, scissors, book   -> the material follows from identity;
                                           the classifier is not consulted.
  constrained  bottle, cup, bowl        -> identity narrows the candidate set,
                                           the classifier picks within it.
  unmapped     person, chair, dog       -> recognised, but not waste.

The constrained case is the important one and the reason this beats either
model alone. A bottle may be plastic, glass or metal, so the object label
cannot decide — but it *can* rule out paper, cardboard, organic and other.
Masking the 7-way softmax to the plausible set can only ever remove wrong
answers, never introduce one, so accuracy is non-decreasing by construction.

What it deliberately does NOT do: resolve plastic vs. glass. That is a genuine
visual ambiguity (a clear PET bottle and a glass bottle are near-identical to a
camera) and is already the classifier's weakest pair. When the constrained
distribution stays close, that is a true "vision cannot settle this" signal —
the case Stage 2's sensors exist to resolve — and it is surfaced as such rather
than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Decision provenance, carried through to the display and the CSV so every
# prediction can be traced back to what actually decided it.
FROM_IDENTITY = "identity"     # object label alone
FROM_CONSTRAINED = "constrained"  # classifier, restricted by object label
FROM_MATERIAL = "material"     # classifier alone (no identity available)


@dataclass(frozen=True)
class Rule:
    """How a recognised COCO class maps onto the canonical waste taxonomy.

    `classes` holds one canonical class for a certain mapping, or the set of
    plausible ones for a constrained mapping. `note` documents the reasoning
    so the table stays auditable when someone disagrees with an entry.
    """

    classes: tuple[str, ...]
    note: str = ""

    @property
    def is_certain(self) -> bool:
        return len(self.classes) == 1


# ---------------------------------------------------------------------------
# COCO class -> waste taxonomy.
#
# Anything absent from this table is treated as "recognised but not waste"
# (person, chair, dog, car, ...) and is dropped rather than force-fitted into
# a class. `Rule.classes` entries must be real edgewaste.taxonomy.CLASS_NAMES
# — this table is built against the 33-item taxonomy, not the flat 7-class
# one it originally targeted (see the note below on what changed).
#
# The 33-class taxonomy is a fine-grained *packaging-waste* taxonomy (nine
# item classes are literally can/bottle/jar variants), not a general
# household-object one, so most COCO objects have no single matching item —
# a fork is recognisably metal, but none of the four metal classes is
# "cutlery". Forcing a fictional exact match would be more misleading than
# admitting the ambiguity, so entries here mostly constrain to a *family*
# (every item in FAMILY_TO_CLASSES[family]) rather than claim certainty they
# do not have. `Rule.classes` is genuinely certain only where a family has
# exactly one item that is the obvious match (a banana is food_waste, not
# eggshells/coffee_grounds/tea_bags; a cell phone is the taxonomy's only
# e-waste item). Two former "other" entries (suitcase, umbrella — no single
# dominant material) are dropped rather than force-fitted, since the
# taxonomy no longer has a catch-all class to put them in at all.
# ---------------------------------------------------------------------------
from .taxonomy import FAMILY_TO_CLASSES as _FAM

COCO_TO_WASTE: dict[str, Rule] = {
    # --- certain: food waste (organic family's one generic-food item) -------
    "banana": Rule(("food_waste",), "food"),
    "apple": Rule(("food_waste",), "food"),
    "orange": Rule(("food_waste",), "food"),
    "broccoli": Rule(("food_waste",), "food"),
    "carrot": Rule(("food_waste",), "food"),
    "sandwich": Rule(("food_waste",), "food"),
    "hot dog": Rule(("food_waste",), "food"),
    "pizza": Rule(("food_waste",), "food"),
    "donut": Rule(("food_waste",), "food"),
    "cake": Rule(("food_waste",), "food"),

    # --- certain: e-waste (hazardous family's one generic electronics item) -
    "cell phone": Rule(("e_waste",), "e-waste"),
    "remote": Rule(("e_waste",), "e-waste"),
    "keyboard": Rule(("e_waste",), "e-waste"),
    "mouse": Rule(("e_waste",), "e-waste"),
    "hair drier": Rule(("e_waste",), "e-waste"),

    # --- constrained: metal family has no cutlery item, but the material
    # itself is unambiguous, so this still narrows the classifier usefully --
    "fork": Rule(_FAM["metal"], "cutlery; no metal item is cutlery-shaped, "
                 "constrains to the material family instead"),
    "knife": Rule(_FAM["metal"], "cutlery; see fork"),
    "spoon": Rule(_FAM["metal"], "cutlery; see fork"),
    "scissors": Rule(_FAM["metal"], "steel blades + plastic handle; metal dominates"),

    # --- constrained: paper/glass families have no exact book/stemware item -
    "book": Rule(_FAM["paper"], "bound printed paper; no item is book-shaped"),
    "wine glass": Rule(_FAM["glass"], "stemware is glass; no item is stemware-shaped"),
    "vase": Rule(_FAM["glass"], "glass or ceramic; ceramic has no home in this "
                 "taxonomy, so this only fires when it plausibly is glass"),

    # --- constrained: textile family (clothing, shoes) ----------------------
    "teddy bear": Rule(_FAM["textile"], "fabric-dominant; closest available family"),
    "tie": Rule(_FAM["textile"], "clothing"),
    "backpack": Rule(_FAM["textile"], "fabric-dominant; closest available family"),
    "handbag": Rule(_FAM["textile"], "fabric-dominant; closest available family"),

    # --- constrained: plastic family, no toothbrush-specific item -----------
    "toothbrush": Rule(_FAM["plastic"], "moulded plastic handle dominates"),

    # --- constrained: identity narrows, classifier decides among survivors --
    "bottle": Rule(("plastic_water_bottles", "plastic_soda_bottles",
                     "plastic_detergent_bottles", "glass_beverage_bottles"),
                    "PET / glass bottle shapes; no metal item is bottle-shaped"),
    "cup": Rule(("paper_cups", "styrofoam_cups", "plastic_cup_lids"),
                "disposable paper, styrofoam, or plastic cup/lid"),
    "bowl": Rule(("plastic_food_containers", "glass_food_jars",
                   "steel_food_cans", "aluminum_food_cans"),
                  "food container across plastic/glass/metal"),
}


@dataclass
class ObjectMatch:
    """A COCO detection, before it is merged with the waste pipeline."""

    box: tuple[int, int, int, int]
    label: str
    conf: float

    @property
    def rule(self) -> Rule | None:
        return COCO_TO_WASTE.get(self.label)

    @property
    def is_waste(self) -> bool:
        return self.label in COCO_TO_WASTE


class ObjectRecognizer:
    """Thin wrapper over a COCO-pretrained YOLO used only for identity.

    Deliberately a *separate* model from the TACO detector rather than a
    replacement: TACO is fine-tuned for cluttered street litter, COCO knows
    everyday objects. Running both costs one extra nano-YOLO pass (~3 ms) and
    each covers the other's blind spot.
    """

    def __init__(self, model: str = "yolo26n.pt", conf_threshold: float = 0.40):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.conf_threshold = float(conf_threshold)
        self.names: dict[int, str] = self.model.names

    def detect(self, image) -> list[ObjectMatch]:
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        w, h = image.size
        out: list[ObjectMatch] = []
        for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                    boxes.cls.tolist()):
            x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
            x2, y2 = min(w, int(xyxy[2])), min(h, int(xyxy[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(ObjectMatch((x1, y1, x2, y2),
                                   self.names[int(cls)], float(conf)))
        return out


def apply_identity_prior(
    probs: np.ndarray,
    rule: Rule | None,
    identity_conf: float,
    name_to_index: dict[str, int],
    certainty: float = 0.95,
) -> tuple[np.ndarray, str]:
    """Fold an object identity into the classifier's distribution.

    Returns `(adjusted_probs, decision_source)`. The result is always a valid
    probability vector over the same 7 classes, so everything downstream —
    temporal smoothing, the confidence gate, the CSV log — keeps working
    untouched. That is the reason this returns a distribution rather than a
    label: the identity prior composes with the smoothing instead of bypassing
    it, so a single mis-recognised frame still cannot flip the display.
    """
    if rule is None:
        return probs, FROM_MATERIAL

    if rule.is_certain:
        # Concentrate mass on the implied class, scaled by how sure the
        # recogniser was. Deliberately NOT a hard one-hot: a 0.42-confidence
        # "banana" should not present as a 100%-certain organic, and leaving
        # residual mass lets smoothing recover if the identity was wrong.
        idx = name_to_index[rule.classes[0]]
        adjusted = np.full_like(probs, 0.0)
        # Two independent sources agreeing must never *lower* confidence. If
        # the classifier already backs the implied class more strongly than
        # the identity does, keep its figure; the prior only ever lifts a
        # class the classifier under-rated.
        peak = max(certainty * identity_conf, float(probs[idx]))
        adjusted[idx] = peak
        others = [i for i in range(len(probs)) if i != idx]
        if others:
            adjusted[others] = (1.0 - peak) / len(others)
        return adjusted, FROM_IDENTITY

    # Constrained: keep only the plausible classes and renormalise. The
    # classifier's relative preference among survivors is preserved exactly —
    # this removes options, it does not re-rank them.
    mask = np.zeros_like(probs)
    for name in rule.classes:
        mask[name_to_index[name]] = 1.0
    masked = probs * mask
    total = masked.sum()
    if total <= 0:
        # Classifier put essentially zero mass on every plausible class — it
        # disagrees with the identity entirely. Fall back to a flat prior over
        # the allowed set rather than dividing by ~0.
        masked = mask / mask.sum()
    else:
        masked = masked / total

    # Renormalising always *inflates* the survivor's confidence, because the
    # competitors it removed had to go somewhere — an 0.86 becomes 0.99 purely
    # by deletion. That is only justified to the extent the identity is
    # trustworthy, so blend back toward the unconstrained distribution in
    # proportion to how sure the recogniser was. A 0.9-confidence "bottle"
    # constrains almost fully; a 0.55-confidence guess barely moves the answer,
    # which is what stops a shaky identity from producing a confident mistake.
    w = float(np.clip(identity_conf, 0.0, 1.0))
    blended = w * masked + (1.0 - w) * probs
    s = blended.sum()
    return (blended / s if s > 0 else masked), FROM_CONSTRAINED


def match_to_boxes(target_box, matches: list[ObjectMatch],
                   iou_threshold: float = 0.45) -> ObjectMatch | None:
    """Best-IoU COCO detection for a given waste box, or None."""
    from .stabilize import iou

    best, best_iou = None, iou_threshold
    for m in matches:
        v = iou(target_box, m.box)
        if v >= best_iou:
            best, best_iou = m, v
    return best
