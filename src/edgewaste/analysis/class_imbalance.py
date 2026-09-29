# class_imbalance.py - class distribution, over-prediction check, and post-hoc logit adjustment for the imbalance correction
"""Class-imbalance analysis.

The 89.30% checkpoint was trained with BOTH a class-balanced sampler and
inverse-frequency loss weights. Each alone makes training behave as if every
class were equally common; together they over-correct, so rare classes are
effectively weighted ~1/n_c relative to frequent ones. Bayes' rule predicts the
consequence: the model's posteriors carry an extra factor of ~1/n_c^2 against
the natural class prior, i.e. it over-predicts rare classes (high recall, low
precision) and under-predicts frequent ones.

Post-hoc logit adjustment (Menon et al., "Long-tail learning via logit
adjustment", ICLR 2021) corrects a known prior mismatch without retraining:

    adjusted_logit_c = logit_c + tau * log(n_c)

tau = 0 is the model as trained; tau = 2 undoes the double correction
exactly under the idealised Bayes argument; tau = 1 approximates what a
single correction (sampler only — the fixed training default) would give.
tau is chosen on the validation split, then applied once to test.
"""

from __future__ import annotations

import numpy as np

from edgewaste.analysis.metrics_report import family_and_hazard


def class_distribution(labels_by_split: dict[str, np.ndarray], class_names: list[str]) -> dict:
    out = {}
    for split, y in labels_by_split.items():
        counts = np.bincount(y, minlength=len(class_names))
        out[split] = {"counts": dict(zip(class_names, counts.tolist())),
                      "max_class": class_names[int(counts.argmax())], "max": int(counts.max()),
                      "min_class": class_names[int(counts.argmin())], "min": int(counts.min()),
                      "imbalance_ratio": float(counts.max() / max(counts.min(), 1))}
    return out


def prediction_share(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """predicted count / true count per class: >1 means the class is over-predicted."""
    return np.bincount(pred, minlength=n_classes) / np.maximum(np.bincount(y, minlength=n_classes), 1)


def adjusted_predictions(logits: np.ndarray, train_counts: np.ndarray, tau: float) -> np.ndarray:
    return (logits + tau * np.log(np.maximum(train_counts, 1))).argmax(1)


def tau_sweep(logits_val, y_val, logits_test, y_test, train_counts, class_names,
              taus=np.round(np.arange(0, 3.01, 0.25), 2)) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    def score(logits, y, tau):
        pred = adjusted_predictions(logits, train_counts, tau)
        return {"tau": float(tau), "accuracy": float((pred == y).mean()),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                **{k: v for k, v in family_and_hazard(y, pred, class_names).items()
                   if k in ("family_accuracy", "hazard_recall", "hazard_precision")}}

    val = [score(logits_val, y_val, t) for t in taus]
    best_tau = max(val, key=lambda r: r["accuracy"])["tau"]
    return {"val": val, "test": [score(logits_test, y_test, t) for t in taus],
            "best_tau_on_val": best_tau}
