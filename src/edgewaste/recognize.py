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
# one of the seven classes. Add entries freely — it is a plain dict, and the
# canonical names must match edgewaste.taxonomy.CLASS_NAMES.
# ---------------------------------------------------------------------------
COCO_TO_WASTE: dict[str, Rule] = {
    # --- certain: food waste -------------------------------------------------
    "banana": Rule(("organic",), "food"),
    "apple": Rule(("organic",), "food"),
    "orange": Rule(("organic",), "food"),
    "broccoli": Rule(("organic",), "food"),
    "carrot": Rule(("organic",), "food"),
    "sandwich": Rule(("organic",), "food"),
    "hot dog": Rule(("organic",), "food"),
    "pizza": Rule(("organic",), "food"),
    "donut": Rule(("organic",), "food"),
    "cake": Rule(("organic",), "food"),

    # --- certain: metal ------------------------------------------------------
    "fork": Rule(("metal",), "cutlery"),
    "knife": Rule(("metal",), "cutlery"),
    "spoon": Rule(("metal",), "cutlery"),
    "scissors": Rule(("metal",), "steel blades + plastic handle; metal dominates"),

    # --- certain: paper / glass ---------------------------------------------
    "book": Rule(("paper",), ""),
    "wine glass": Rule(("glass",), "stemware is glass by definition"),

    # --- certain: other (e-waste, textile, misc non-recyclable) --------------
    "cell phone": Rule(("other",), "e-waste"),
    "remote": Rule(("other",), "e-waste"),
    "keyboard": Rule(("other",), "e-waste"),
    "mouse": Rule(("other",), "e-waste"),
    "hair drier": Rule(("other",), "e-waste"),
    "teddy bear": Rule(("other",), "textile"),
    "tie": Rule(("other",), "textile"),
    "backpack": Rule(("other",), "textile"),
    "handbag": Rule(("other",), "textile"),
    "suitcase": Rule(("other",), "mixed materials"),
    "umbrella": Rule(("other",), "mixed materials"),
    "toothbrush": Rule(("plastic",), "moulded plastic handle dominates"),

    # --- constrained: identity narrows, classifier decides -------------------
    "bottle": Rule(("plastic", "glass", "metal"),
                   "PET / glass / aluminium all present as bottles"),
    "cup": Rule(("paper", "plastic", "glass"),
                "disposable paper, plastic, or a glass tumbler"),
    "bowl": Rule(("plastic", "glass", "metal", "paper"),
                 "widest container ambiguity — rules out only organic/cardboard"),
    "vase": Rule(("glass", "other"), "glass or ceramic; ceramic falls to other"),
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
