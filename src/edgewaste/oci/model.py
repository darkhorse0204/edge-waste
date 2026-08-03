"""The deployed OCI model, its offline fitting procedure, the required
ablation study, and safety-leaning threshold selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OCIModel:
    """Three fitted logistic regressions, not one: full (both sensors),
    moisture-only, gas-only. Selecting the right dedicated model when a
    sensor drops out (compute_oci below) produces far fewer classification
    flips than algebraically rescaling the combined model's terms."""
    beta0: float
    beta1: float
    beta2: float
    beta3: float = 0.0
    m_only_beta0: float = 0.0
    m_only_beta1: float = 0.0
    g_only_beta0: float = 0.0
    g_only_beta1: float = 0.0
    operating_threshold: float = 0.5


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_oci(model: OCIModel, f_m: float | None, f_g: float | None) -> float:
    if f_m is not None and f_g is not None:
        linear = model.beta0 + model.beta1 * f_m + model.beta2 * f_g
        if model.beta3:
            linear += model.beta3 * f_m * f_g
        return _sigmoid(linear)
    if f_m is not None:
        return _sigmoid(model.m_only_beta0 + model.m_only_beta1 * f_m)
    if f_g is not None:
        return _sigmoid(model.g_only_beta0 + model.g_only_beta1 * f_g)
    raise RuntimeError("Both sensors unavailable — route to manual review.")


def fit_oci_weights(f_m: np.ndarray, f_g: np.ndarray, y: np.ndarray,
                     use_interaction: bool = False, C: float = 1.0
                     ) -> tuple[OCIModel, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    corr = float(np.corrcoef(f_m, f_g)[0, 1])

    X = np.column_stack([f_m, f_g] + ([f_m * f_g] if use_interaction else []))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    clf = LogisticRegression(C=C).fit(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    clf_m = LogisticRegression(C=C).fit(f_m.reshape(-1, 1), y)
    clf_g = LogisticRegression(C=C).fit(f_g.reshape(-1, 1), y)

    model = OCIModel(
        beta0=float(clf.intercept_[0]),
        beta1=float(clf.coef_[0][0]),
        beta2=float(clf.coef_[0][1]),
        beta3=float(clf.coef_[0][2]) if use_interaction else 0.0,
        m_only_beta0=float(clf_m.intercept_[0]),
        m_only_beta1=float(clf_m.coef_[0][0]),
        g_only_beta0=float(clf_g.intercept_[0]),
        g_only_beta1=float(clf_g.coef_[0][0]),
    )
    diagnostics = {
        "moisture_gas_correlation": corr,
        "test_accuracy": test_acc,
        "n_train": len(y_train), "n_test": len(y_test),
        "regularization_C": C, "used_interaction": use_interaction,
    }
    return model, diagnostics


def run_ablation(f_m: np.ndarray, f_g: np.ndarray, y: np.ndarray,
                  n_splits: int = 5) -> dict:
    """5-fold cross-validated AUC. With only ~80-160 calibration samples, a
    single 80/20 split's test set is small enough that AUC swings heavily
    on which samples land in it -- CV averages over multiple splits."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def _cv_auc(X):
        return float(cross_val_score(
            LogisticRegression(), X, y, cv=cv, scoring="roc_auc").mean())

    return {
        "moisture_only_auc": _cv_auc(f_m.reshape(-1, 1)),
        "gas_only_auc": _cv_auc(f_g.reshape(-1, 1)),
        "combined_auc": _cv_auc(np.column_stack([f_m, f_g])),
        "cv_folds": n_splits,
    }


def select_threshold(y_true: np.ndarray, y_scores: np.ndarray,
                      min_sensitivity: float = 0.95) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = float(roc_auc_score(y_true, y_scores))
    valid = np.where(tpr >= min_sensitivity)[0]
    if len(valid) == 0:
        raise RuntimeError(
            f"No threshold achieves {min_sensitivity:.0%} sensitivity on this "
            "data — collect more calibration samples or relax the target.")
    best_idx = valid[np.argmin(fpr[valid])]
    return {
        "operating_threshold": float(thresholds[best_idx]),
        "achieved_sensitivity": float(tpr[best_idx]),
        "achieved_fpr": float(fpr[best_idx]),
        "roc_auc": auc,
    }