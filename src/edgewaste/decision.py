"""Decision engine (blueprint Module 9) — combines material class, OCI
contamination score, and MC-Dropout uncertainty into one segregation
decision, then a mock conveyor/servo actuator carries it out.

No physical conveyor exists, so `actuate_conveyor` logs the action it would
take (which gate/bin, in what direction) instead of driving GPIO. Swap its
body for real servo control the moment hardware exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import FAMILIES, FAMILY_TO_INDEX, family_of, is_hazardous

# A physical sorter has one gate per *material family*, not per item class:
# a water bottle and a soda bottle go down the same chute. So gates are
# indexed by family, plus two extra gates for items the vision model is
# unsure about or that OCI flags as contaminated.
GATE_INDEX: dict[str, int] = dict(FAMILY_TO_INDEX)
GATE_MANUAL_REVIEW = len(FAMILIES)
GATE_CONTAMINATED_REJECT = len(FAMILIES) + 1

UNCERTAINTY_REVIEW_THRESHOLD = 0.5  # normalised predictive entropy
OCI_CONTAMINATION_THRESHOLD = 0.5  # sigmoid score; overridden by select_threshold() when calibrated


@dataclass
class Decision:
    class_name: str  # fine-grained item class
    family: str  # material family it rolls up into
    cls_conf: float
    uncertainty: float
    oci_score: float | None
    route: str  # a material family, "manual_review", or "contaminated_reject"
    gate: int
    reason: str


def decide(
    class_name: str, cls_conf: float, uncertainty: float, oci_score: float | None,
    uncertainty_threshold: float = UNCERTAINTY_REVIEW_THRESHOLD,
    oci_threshold: float = OCI_CONTAMINATION_THRESHOLD,
) -> Decision:
    """Route one item. Ambiguity beats confidence: an uncertain vision call
    is checked by a human before an OCI-contaminated but confidently-typed
    recyclable is rejected, since a misrouted "unsure" item is more costly
    than a conservative contamination reject.

    Hazard outranks both. A suspected battery, e-waste or medical item goes
    to the hazardous gate even when the model is unsure, because the cost of
    a missed hazard (fire, injury) is far above the cost of a human checking
    a false alarm.

    Routing is by material *family*, not item class: a water bottle and a
    soda bottle share a chute, so a fine-grained confusion within a family
    is harmless at this stage. That is what makes the hierarchical metric in
    `evaluate.py` the one that reflects real sorting performance.
    """
    family = family_of(class_name)

    if is_hazardous(class_name):
        return Decision(class_name, family, cls_conf, uncertainty, oci_score,
                         route="hazardous", gate=GATE_INDEX["hazardous"],
                         reason=f"'{class_name}' is a hazardous stream - "
                                f"never routed to recycling or compost")

    if uncertainty >= uncertainty_threshold:
        return Decision(class_name, family, cls_conf, uncertainty, oci_score,
                         route="manual_review", gate=GATE_MANUAL_REVIEW,
                         reason=f"uncertainty {uncertainty:.2f} >= {uncertainty_threshold:.2f}")

    if oci_score is not None and oci_score >= oci_threshold and family != "organic":
        return Decision(class_name, family, cls_conf, uncertainty, oci_score,
                         route="contaminated_reject", gate=GATE_CONTAMINATED_REJECT,
                         reason=f"OCI {oci_score:.2f} >= {oci_threshold:.2f} on a "
                                f"non-organic item - recycling-stream contamination risk")

    return Decision(class_name, family, cls_conf, uncertainty, oci_score,
                     route=family, gate=GATE_INDEX[family],
                     reason="clean classification, low uncertainty, OCI below threshold")


def actuate_conveyor(decision: Decision) -> str:
    """Mock conveyor/servo actuation. Returns the human-readable action log
    line; a real deployment would drive a GPIO servo to gate `decision.gate`
    instead of just logging it."""
    msg = (f"[CONVEYOR] gate={decision.gate:<2} route={decision.route:<20} "
           f"item={decision.class_name:<28} conf={decision.cls_conf*100:5.1f}%  "
           f"({decision.reason})")
    print(msg)
    return msg
