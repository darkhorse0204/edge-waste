"""Run the full OCI pipeline end-to-end on synthetic data. Confirms
normalize.py/model.py work correctly BEFORE real sensor hardware arrives —
swap generate_synthetic_calibration_data() for a real calibration_data.csv
load once you have one, nothing else in this script should need to change."""

import numpy as np

from edgewaste.oci import compute_oci, fit_oci_weights, run_ablation, select_threshold
from edgewaste.oci.normalize import normalize_gas, normalize_moisture
from edgewaste.oci.synthetic import generate_synthetic_calibration_data

df, anchors = generate_synthetic_calibration_data()

df["f_m"] = df["moisture_raw"].apply(lambda m: normalize_moisture(m, anchors))
df["f_g"] = df["rs_over_r0"].apply(lambda r: normalize_gas(r, anchors))

f_m = df["f_m"].to_numpy()
f_g = df["f_g"].to_numpy()
y = df["contaminated"].to_numpy()

coeffs, diagnostics = fit_oci_weights(f_m, f_g, y)
print("Fitted coefficients:", coeffs)
print("Diagnostics:", diagnostics)

ablation = run_ablation(f_m, f_g, y)
print("Ablation (AUC):", ablation)

y_scores = np.array([compute_oci(coeffs, m, g) for m, g in zip(f_m, f_g)])
threshold_info = select_threshold(y, y_scores, min_sensitivity=0.95)
print("Threshold selection:", threshold_info)

# Sensor-dropout check — same items, moisture only vs. gas only vs. both.
print("Sample with both sensors:  ", compute_oci(coeffs, f_m[0], f_g[0]))
print("Sample, moisture only:     ", compute_oci(coeffs, f_m[0], None))
print("Sample, gas only:          ", compute_oci(coeffs, None, f_g[0]))