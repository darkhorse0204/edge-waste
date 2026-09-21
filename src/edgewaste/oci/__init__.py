"""Organic Contamination Index (OCI) — Stage 2's sensor-fusion novelty claim.

Fuses a capacitive moisture reading and an MQ-135 gas reading into a single
0-1 contamination score via fixed-reference normalisation + logistic
regression. See `normalize.py` for the feature scaling, `model.py` for the
fitted model and its fitting/ablation/thresholding procedures, and
`synthetic.py` for a physically-motivated fake calibration dataset (stand-in
until real 160-sample hardware calibration data exists).
"""

from .model import OCIModel, compute_oci, fit_oci_weights, run_ablation, select_threshold
from .normalize import CalibrationAnchors, normalize_gas, normalize_moisture

__all__ = [
    "CalibrationAnchors",
    "normalize_moisture",
    "normalize_gas",
    "OCIModel",
    "compute_oci",
    "fit_oci_weights",
    "run_ablation",
    "select_threshold",
]
