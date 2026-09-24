"""Risk-bounded routing: family-conditional (Mondrian) split-conformal
prediction over the classifier's *family-marginalised* probabilities.

Why this exists. `decide()` only treats an item as hazardous when the
classifier's single top item class is a hazard class. A battery whose top
guess is `office_paper` (the most common miss on the held-out test set) goes
straight to a recycling gate, and nothing in the system bounds how often that
happens — the hazard recall number is whatever the model happens to achieve.

This module turns that into a user-set, finite-sample guarantee:

  1. Collapse the 33 item probabilities into 9 family masses
     m_f(x) = sum_{i in family f} p_i(x). Summing first matters: a battery
     split 30/25/20 across battery/e_waste/medical never wins argmax but has
     75% hazardous mass.
  2. For each family f separately, calibrate a threshold q_f on held-out
     items whose true family is f, using nonconformity s = 1 - m_f(x).
  3. The prediction set is C(x) = {f : 1 - m_f(x) <= q_f}. By split-conformal
     exchangeability, for a new item of true family f:
         P(f in C(X) | Y = f) >= 1 - alpha_f      (finite-sample, any model)
  4. Route from the set, not from argmax:
       hazardous in C, |C| = 1  -> hazardous actuation
       hazardous in C, |C| > 1  -> hazard-suspected manual review
       hazardous not in C, |C| = 1 -> automatic routing to that family's gate
       otherwise (empty or ambiguous) -> manual review

Consequences, each following from (3) alone:
  * A true hazard reaches any non-hazardous gate only if hazardous is not in
    C(x): probability <= alpha_hazard.
  * An item of true family f is auto-actuated into a wrong gate only if
    f is not in C(x): probability <= alpha_f.

The guarantee assumes calibration and deployment items are exchangeable, so it
says nothing about out-of-distribution input. That is what the MC-Dropout
uncertainty gate in `decide()` remains for; the two gates are complementary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .taxonomy import CLASS_TO_FAMILY, FAMILIES, FAMILY_TO_INDEX

HAZARD_FAMILY = "hazardous"
HAZ = FAMILY_TO_INDEX[HAZARD_FAMILY]

ROUTE_AUTO = "auto"
ROUTE_HAZARD = "hazardous"
ROUTE_HAZARD_REVIEW = "hazard_review"
ROUTE_REVIEW = "review"


def family_matrix(class_names: list[str]) -> np.ndarray:
    """[num_classes, num_families] 0/1 matrix mapping items to families."""
    m = np.zeros((len(class_names), len(FAMILIES)))
    for i, name in enumerate(class_names):
        m[i, FAMILY_TO_INDEX[CLASS_TO_FAMILY[name]]] = 1.0
    return m


def family_mass(probs: np.ndarray, class_names: list[str]) -> np.ndarray:
    return probs @ family_matrix(class_names)


def conformal_rank(n: int, alpha: float, delta: float | None = None) -> int | None:
    """1-based rank of the calibration score to use as threshold, or None if
    n is too small to certify the target.

    delta=None: standard split conformal, coverage >= 1-alpha on average over
    calibration draws (a single deployment can land somewhat below it).

    delta given: calibration-conditional (PAC) version. The true coverage of
    the k-th smallest of n scores is Beta(k, n+1-k) distributed, so choosing
    the smallest k whose delta-quantile clears 1-alpha gives coverage >= 1-alpha
    with probability >= 1-delta for THIS calibration set — a per-deployment
    statement rather than an average one."""
    if n == 0:
        return None
    if delta is None:
        k = math.ceil((n + 1) * (1 - alpha))
        return k if k <= n else None
    from scipy.stats import beta
    ks = np.arange(1, n + 1)
    ok = beta.ppf(delta, ks, n + 1 - ks) >= 1 - alpha
    return int(ks[ok][0]) if ok.any() else None


def conformal_quantile(scores: np.ndarray, alpha: float, delta: float | None = None) -> float:
    """Threshold = score at conformal_rank. +inf (family always kept in the
    set) when n is too small to certify the target — the conservative
    failure mode."""
    k = conformal_rank(len(scores), alpha, delta)
    return math.inf if k is None else float(np.sort(scores)[k - 1])


def alpha_vector(alpha: float, alpha_hazard: float) -> np.ndarray:
    a = np.full(len(FAMILIES), alpha)
    a[HAZ] = alpha_hazard
    return a


def calibrate(fam_mass: np.ndarray, fam_true: np.ndarray, alphas: np.ndarray,
              delta: float | None = None) -> np.ndarray:
    return np.array([
        conformal_quantile(1.0 - fam_mass[fam_true == f, f], alphas[f], delta)
        for f in range(len(FAMILIES))
    ])


def prediction_sets(fam_mass: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (1.0 - fam_mass) <= q


def route_sets(sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (route kind per item, routed family index or -1)."""
    size = sets.sum(axis=1)
    haz_in = sets[:, HAZ]
    kind = np.full(len(sets), ROUTE_REVIEW, dtype=object)
    kind[haz_in & (size == 1)] = ROUTE_HAZARD
    kind[haz_in & (size > 1)] = ROUTE_HAZARD_REVIEW
    kind[~haz_in & (size == 1)] = ROUTE_AUTO
    fam = np.where(size == 1, sets.argmax(axis=1), -1)
    return kind, fam


@dataclass
class ConformalRouter:
    thresholds: np.ndarray  # q_f per family
    alphas: np.ndarray
    delta: float | None = None

    @classmethod
    def fit(cls, probs: np.ndarray, labels: np.ndarray, class_names: list[str],
            alpha: float = 0.05, alpha_hazard: float = 0.02,
            delta: float | None = 0.05) -> "ConformalRouter":
        fm = family_mass(probs, class_names)
        fam_true = family_matrix(class_names)[labels].argmax(axis=1)
        alphas = alpha_vector(alpha, alpha_hazard)
        return cls(calibrate(fm, fam_true, alphas, delta), alphas, delta)

    def family_set(self, item_probs: np.ndarray, class_names: list[str]) -> set[str]:
        fm = family_mass(item_probs[None, :], class_names)
        members = prediction_sets(fm, self.thresholds)[0]
        return {FAMILIES[f] for f in np.flatnonzero(members)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "families": list(FAMILIES),
            "thresholds": [None if math.isinf(t) else t for t in self.thresholds],
            "alphas": self.alphas.tolist(),
            "delta": self.delta,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ConformalRouter":
        d = json.loads(Path(path).read_text())
        if list(d["families"]) != list(FAMILIES):
            raise ValueError("calibration file was fitted on a different family taxonomy")
        q = np.array([math.inf if t is None else t for t in d["thresholds"]])
        return cls(q, np.array(d["alphas"]), d.get("delta"))
