# routing_rules.py - final routing: family bin, hazardous stream, contamination reject or manual review
"""Decision engine (blueprint Module 9) — combines material class, OCI
contamination score, and MC-Dropout uncertainty into one segregation
decision, then a mock conveyor/servo actuator carries it out.

No physical conveyor exists, so `actuate_conveyor` logs the action it would
take (which gate/bin, in what direction) instead of driving GPIO. Swap its
body for real servo control the moment hardware exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from edgewaste.taxonomy import FAMILIES, FAMILY_TO_INDEX, family_of, is_hazardous

# a physical sorter has one gate per *material family*, not per item class:
# a water bottle and a soda bottle go down the same chute. so gates are
# indexed by family, plus two extra gates for items the vision model is
# unsure about or that oci flags as contaminated.
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
    route: str  # a material family, "manual_review", "contaminated_reject", or "hazardous"
    gate: int
    reason: str
    hazard_suspected: bool = False  # classifier named a hazard class, but too unsure to act on it directly


def decide(
    class_name: str, cls_conf: float, uncertainty: float, oci_score: float | None,
    uncertainty_threshold: float = UNCERTAINTY_REVIEW_THRESHOLD,
    oci_threshold: float = OCI_CONTAMINATION_THRESHOLD,
    family_set: set[str] | None = None,
) -> Decision:
    """Route one item. Ambiguity beats confidence: an uncertain vision call
    is checked by a human before an OCI-contaminated but confidently-typed
    recyclable is rejected, since a misrouted "unsure" item is more costly
    than a conservative contamination reject.

    Hazard beats ambiguity, but not blindly. A *confident* battery/e-waste/
    medical call skips the uncertainty check entirely and gets acted on
    immediately — the cost of a missed hazard (fire, injury) outweighs the
    cost of a human checking a false alarm. But when the classifier is both
    uncertain AND happened to land on a hazard class, that combination is not
    a confirmed hazard, it is noise that landed on the scariest label by
    chance — observed directly on out-of-distribution crops, where a
    low-confidence classifier disproportionately guessed hazard classes and
    would have fired real hazardous-stream actuation on video frames that
    were, physically, cardboard and cups. So an uncertain hazard call still
    never reaches a recycling or compost gate (the one guarantee that must
    hold unconditionally), but goes to manual review flagged as
    hazard-suspected rather than being acted on automatically.

    Routing is by material *family*, not item class: a water bottle and a
    soda bottle share a chute, so a fine-grained confusion within a family
    is harmless at this stage. That is what makes the hierarchical metric in
    `classification/evaluate_classifier.py` the one that reflects real sorting performance.

    `family_set`, when given, is the conformal prediction set from
    `conformal.ConformalRouter`; routing then follows the set instead of the
    top-1 class, which bounds hazard leakage at a calibrated rate.
    """
    if family_set is not None:
        return _decide_from_set(class_name, cls_conf, uncertainty, oci_score,
                                family_set, uncertainty_threshold, oci_threshold)

    family = family_of(class_name)

    if is_hazardous(class_name):
        if uncertainty >= uncertainty_threshold:
            return Decision(class_name, family, cls_conf, uncertainty, oci_score,
                             route="manual_review", gate=GATE_MANUAL_REVIEW,
                             hazard_suspected=True,
                             reason=f"possible '{class_name}' but uncertainty "
                                    f"{uncertainty:.2f} >= {uncertainty_threshold:.2f} - "
                                    f"too unsure to act on automatically, priority human check")
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


def _decide_from_set(
    class_name: str, cls_conf: float, uncertainty: float, oci_score: float | None,
    family_set: set[str], uncertainty_threshold: float, oci_threshold: float,
) -> Decision:
    """Set-valued routing. Any set containing the hazardous family never
    reaches a recycling/compost gate; automatic actuation needs a singleton
    set AND low MC-Dropout uncertainty (the conformal guarantee does not
    cover out-of-distribution input, the uncertainty gate does)."""
    ordered = sorted(family_set, key=FAMILY_TO_INDEX.get)
    family = ordered[0] if len(ordered) == 1 else family_of(class_name)
    set_txt = "{" + ", ".join(ordered) + "}"

    if "hazardous" in family_set:
        if len(family_set) == 1 and uncertainty < uncertainty_threshold:
            return Decision(class_name, "hazardous", cls_conf, uncertainty, oci_score,
                             route="hazardous", gate=GATE_INDEX["hazardous"],
                             reason=f"conformal set {set_txt} - hazardous stream")
        return Decision(class_name, family, cls_conf, uncertainty, oci_score,
                         route="manual_review", gate=GATE_MANUAL_REVIEW,
                         hazard_suspected=True,
                         reason=f"conformal set {set_txt} includes hazardous - "
                                f"barred from recycling, priority human check")

    if len(family_set) != 1 or uncertainty >= uncertainty_threshold:
        return Decision(class_name, family, cls_conf, uncertainty, oci_score,
                         route="manual_review", gate=GATE_MANUAL_REVIEW,
                         reason=f"conformal set {set_txt}, uncertainty {uncertainty:.2f}")

    if oci_score is not None and oci_score >= oci_threshold and family != "organic":
        return Decision(class_name, family, cls_conf, uncertainty, oci_score,
                         route="contaminated_reject", gate=GATE_CONTAMINATED_REJECT,
                         reason=f"OCI {oci_score:.2f} >= {oci_threshold:.2f} on a "
                                f"non-organic item - recycling-stream contamination risk")

    return Decision(class_name, family, cls_conf, uncertainty, oci_score,
                     route=family, gate=GATE_INDEX[family],
                     reason=f"conformal singleton {set_txt}, low uncertainty, OCI below threshold")


def actuate_conveyor(decision: Decision) -> str:
    """Mock conveyor/servo actuation. Returns the human-readable action log
    line; a real deployment would drive a GPIO servo to gate `decision.gate`
    instead of just logging it."""
    flag = " [HAZARD SUSPECTED]" if decision.hazard_suspected else ""
    msg = (f"[CONVEYOR] gate={decision.gate:<2} route={decision.route:<20} "
           f"item={decision.class_name:<28} conf={decision.cls_conf*100:5.1f}%{flag}  "
           f"({decision.reason})")
    print(msg)
    return msg
