"""Decision engine (blueprint Module 9) — combines material class, OCI
contamination score, and MC-Dropout uncertainty into one segregation
decision, then a mock conveyor/servo actuator carries it out.

No physical conveyor exists, so `actuate_conveyor` logs the action it would
take (which gate/bin, in what direction) instead of driving GPIO. Swap its
body for real servo control the moment hardware exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import CLASS_NAMES

# Gate index per canonical class, plus two extra gates for items the vision
# model is unsure about or that OCI flags as contaminated.
GATE_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
GATE_MANUAL_REVIEW = len(CLASS_NAMES)
GATE_CONTAMINATED_REJECT = len(CLASS_NAMES) + 1
GATE_HAZARDOUS = len(CLASS_NAMES) + 2

# Streams that must never reach a recycling or compost gate: a battery in a
# paper bale is a fire, a syringe in a sorting line is an injury. These are
# routed on class alone, before any contamination or confidence logic.
HAZARDOUS_CLASSES = frozenset({"battery", "e_waste", "medical"})

UNCERTAINTY_REVIEW_THRESHOLD = 0.5  # normalised predictive entropy
OCI_CONTAMINATION_THRESHOLD = 0.5  # sigmoid score; overridden by select_threshold() when calibrated


@dataclass
class Decision:
    class_name: str
    cls_conf: float
    uncertainty: float
    oci_score: float | None
    route: str  # canonical class name, "manual_review", or "contaminated_reject"
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
    """
    if class_name in HAZARDOUS_CLASSES:
        return Decision(class_name, cls_conf, uncertainty, oci_score,
                         route="hazardous", gate=GATE_HAZARDOUS,
                         reason=f"'{class_name}' is a hazardous stream - "
                                f"never routed to recycling or compost")

    if uncertainty >= uncertainty_threshold:
        return Decision(class_name, cls_conf, uncertainty, oci_score,
                         route="manual_review", gate=GATE_MANUAL_REVIEW,
                         reason=f"uncertainty {uncertainty:.2f} >= {uncertainty_threshold:.2f}")

    if oci_score is not None and oci_score >= oci_threshold and class_name != "organic":
        return Decision(class_name, cls_conf, uncertainty, oci_score,
                         route="contaminated_reject", gate=GATE_CONTAMINATED_REJECT,
                         reason=f"OCI {oci_score:.2f} >= {oci_threshold:.2f} on a "
                                f"non-organic item - recycling-stream contamination risk")

    return Decision(class_name, cls_conf, uncertainty, oci_score,
                     route=class_name, gate=GATE_INDEX.get(class_name, GATE_MANUAL_REVIEW),
                     reason="clean classification, low uncertainty, OCI below threshold")


def actuate_conveyor(decision: Decision) -> str:
    """Mock conveyor/servo actuation. Returns the human-readable action log
    line; a real deployment would drive a GPIO servo to gate `decision.gate`
    instead of just logging it."""
    msg = (f"[CONVEYOR] gate={decision.gate:<2} route={decision.route:<20} "
           f"class={decision.class_name:<10} conf={decision.cls_conf*100:5.1f}%  "
           f"({decision.reason})")
    print(msg)
    return msg
