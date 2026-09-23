"""SHAP feature attribution for the OCI logistic regression.

OCI's combined model is `sigmoid(beta0 + beta1*f_m + beta2*f_g [+ beta3*f_m*f_g])`
— a two- or three-feature linear model in logit space. SHAP's LinearExplainer
is exact (not approximated) for a linear model given its coefficients and a
background distribution, so there is no reason to reach for a sampling-based
explainer here; KernelExplainer would spend compute approximating a value
LinearExplainer computes in closed form.

The interaction term, when present, is folded in as a third explicit feature
(f_m * f_g) rather than left for SHAP to infer, since LinearExplainer explains
exactly the linear model it is given — an interaction the model uses but the
explainer doesn't see would attribute its effect to the wrong features.
"""

from __future__ import annotations

import numpy as np

from .model import OCIModel


def _feature_matrix(f_m: np.ndarray, f_g: np.ndarray, use_interaction: bool) -> np.ndarray:
    cols = [f_m, f_g]
    if use_interaction:
        cols.append(f_m * f_g)
    return np.column_stack(cols)


def shap_explain_oci(
    model: OCIModel, f_m: np.ndarray, f_g: np.ndarray,
    f_m_background: np.ndarray, f_g_background: np.ndarray,
) -> dict:
    """Per-sample SHAP values for the combined (both-sensor) OCI model.

    `f_m`/`f_g` are the normalised readings to explain (one or many samples);
    `f_m_background`/`f_g_background` set the reference distribution SHAP
    attributes deviations *from* — normally the calibration dataset.

    Returns base_value (the model's output on an average-background sample)
    and per-feature SHAP values that sum to (prediction - base_value) for
    every sample, which is the identity that makes an attribution a genuine
    decomposition rather than a plausible-looking heuristic.
    """
    import shap

    use_interaction = bool(model.beta3)
    coefs = np.array(
        [model.beta1, model.beta2] + ([model.beta3] if use_interaction else [])
    )

    background = _feature_matrix(
        np.asarray(f_m_background, dtype=float),
        np.asarray(f_g_background, dtype=float),
        use_interaction,
    )
    X = _feature_matrix(
        np.atleast_1d(np.asarray(f_m, dtype=float)),
        np.atleast_1d(np.asarray(f_g, dtype=float)),
        use_interaction,
    )

    # A linear model in the *logit*: SHAP explains that linear function
    # (base_value + beta0 offset), the sigmoid is then applied to interpret
    # magnitudes but does not change which feature gets credit for what.
    explainer = shap.LinearExplainer(
        (coefs, model.beta0), background, feature_names=None
    )
    shap_values = explainer.shap_values(X)

    feature_names = ["moisture", "gas"] + (["moisture_x_gas"] if use_interaction else [])
    return {
        "feature_names": feature_names,
        "base_value": float(explainer.expected_value),
        "shap_values": shap_values.tolist(),
        "sanity_check": "base_value + sum(shap_values[i]) == logit(prediction[i])",
    }


def explain_one(model: OCIModel, f_m: float, f_g: float,
                 f_m_background: np.ndarray, f_g_background: np.ndarray) -> str:
    """Human-readable one-item explanation, for logs and the report."""
    result = shap_explain_oci(model, np.array([f_m]), np.array([f_g]),
                               f_m_background, f_g_background)
    sv = result["shap_values"][0]
    parts = [f"{name}={val:+.3f}" for name, val in zip(result["feature_names"], sv)]
    return (f"OCI SHAP: base={result['base_value']:.3f}  "
            + "  ".join(parts))
