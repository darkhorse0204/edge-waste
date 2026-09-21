"""The fitted OCI model and its fitting / ablation / thresholding procedures.

Three separate logistic regressions are fitted, not one: a combined
two-sensor model, a moisture-only model, and a gas-only model. When a sensor
drops out at inference time, `compute_oci` selects the dedicated
single-sensor model instead of algebraically rescaling the combined one —
this produces far fewer classification flips than pretending a missing
feature is zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OCIModel:
    """Coefficients for all three fitted logistic regressions."""

    # Combined model: sigmoid(beta0 + beta1*f_m + beta2*f_g [+ beta3*f_m*f_g])
    beta0: float
    beta1_moisture: float
    beta2_gas: float
    beta3_interaction: float = 0.0
    # Moisture-only fallback: sigmoid(beta0_m + beta1_m*f_m)
    beta0_m: float = 0.0
    beta1_m: float = 0.0
    # Gas-only fallback: sigmoid(beta0_g + beta1_g*f_g)
    beta0_g: float = 0.0
    beta1_g: float = 0.0
    # Whether the combined model includes the interaction term.
    use_interaction: bool = False


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def compute_oci(
    model: OCIModel, f_m: float | None = None, f_g: float | None = None
) -> float:
    """Sigmoid over whichever fitted model matches sensor availability.

    Raises RuntimeError if both sensors are dead — route to manual review
    rather than guessing.
    """
    if f_m is not None and f_g is not None:
        z = model.beta0 + model.beta1_moisture * f_m + model.beta2_gas * f_g
        if model.use_interaction:
            z += model.beta3_interaction * f_m * f_g
        return _sigmoid(z)
    if f_m is not None:
        return _sigmoid(model.beta0_m + model.beta1_m * f_m)
    if f_g is not None:
        return _sigmoid(model.beta0_g + model.beta1_g * f_g)
    raise RuntimeError(
        "Both moisture and gas sensors unavailable — cannot compute OCI. "
        "Route this item to manual review."
    )


def fit_oci_weights(
    f_m: np.ndarray, f_g: np.ndarray, y: np.ndarray, use_interaction: bool = False
) -> tuple[OCIModel, dict]:
    """Fit the combined + two single-sensor logistic regressions.

    Returns (model, diagnostics) where diagnostics includes the moisture/gas
    Pearson correlation — the collinearity check the calibration protocol
    demands (near-perfect correlation would make the ablation meaningless).
    """
    from sklearn.linear_model import LogisticRegression

    f_m = np.asarray(f_m, dtype=float)
    f_g = np.asarray(f_g, dtype=float)
    y = np.asarray(y, dtype=int)

    X_combined = np.column_stack([f_m, f_g] + ([f_m * f_g] if use_interaction else []))
    clf_combined = LogisticRegression().fit(X_combined, y)
    clf_m = LogisticRegression().fit(f_m.reshape(-1, 1), y)
    clf_g = LogisticRegression().fit(f_g.reshape(-1, 1), y)

    b0, b1, b2 = float(clf_combined.intercept_[0]), *[float(c) for c in clf_combined.coef_[0][:2]]
    b3 = float(clf_combined.coef_[0][2]) if use_interaction else 0.0

    model = OCIModel(
        beta0=b0,
        beta1_moisture=b1,
        beta2_gas=b2,
        beta3_interaction=b3,
        beta0_m=float(clf_m.intercept_[0]),
        beta1_m=float(clf_m.coef_[0][0]),
        beta0_g=float(clf_g.intercept_[0]),
        beta1_g=float(clf_g.coef_[0][0]),
        use_interaction=use_interaction,
    )
    pearson_r = float(np.corrcoef(f_m, f_g)[0, 1]) if len(f_m) > 1 else float("nan")
    diagnostics = {
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "moisture_gas_pearson_r": pearson_r,
    }
    return model, diagnostics


def run_ablation(f_m: np.ndarray, f_g: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """5-fold stratified CV AUC for moisture-only, gas-only, and combined.

    A single 80/20 split of ~160 samples swings wildly, so cross-validation
    is used explicitly rather than a single held-out split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    f_m = np.asarray(f_m, dtype=float)
    f_g = np.asarray(f_g, dtype=float)
    y = np.asarray(y, dtype=int)

    variants = {
        "moisture_only": np.column_stack([f_m]),
        "gas_only": np.column_stack([f_g]),
        "combined": np.column_stack([f_m, f_g]),
    }
    results: dict[str, list[float]] = {k: [] for k in variants}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for name, X in variants.items():
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression().fit(X[train_idx], y[train_idx])
            scores = clf.predict_proba(X[test_idx])[:, 1]
            if len(np.unique(y[test_idx])) < 2:
                continue  # a fold with only one class has no defined AUC
            results[name].append(roc_auc_score(y[test_idx], scores))
    return {
        name: {"mean_auc": float(np.mean(v)), "std_auc": float(np.std(v)), "n_folds": len(v)}
        for name, v in results.items()
    }


def select_threshold(y_true: np.ndarray, y_score: np.ndarray, min_sensitivity: float = 0.95) -> dict:
    """Minimum-FPR threshold among those achieving >= min_sensitivity (TPR).

    A missed contamination is worse than an unnecessary reject, so sensitivity
    is a hard constraint, not just one term in Youden's J. Raises if no
    threshold in the ROC curve reaches the sensitivity floor.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    eligible = tpr >= min_sensitivity
    if not eligible.any():
        raise ValueError(
            f"No threshold on this ROC curve reaches {min_sensitivity:.0%} sensitivity; "
            "need more/better calibration data."
        )
    idx = np.argmin(np.where(eligible, fpr, np.inf))
    return {
        "threshold": float(thresholds[idx]),
        "sensitivity": float(tpr[idx]),
        "fpr": float(fpr[idx]),
    }
