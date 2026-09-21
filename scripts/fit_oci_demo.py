"""End-to-end OCI demo: normalise -> fit -> ablate -> select threshold ->
compute OCI under all three sensor-availability cases.

Runs on synthetic calibration data by default. Once a real
`calibration_data.csv` exists (columns: category, severity_g, moisture_raw,
rs, contaminated), pass --csv and nothing else in this script changes.

Usage:
    python scripts/fit_oci_demo.py
    python scripts/fit_oci_demo.py --csv path/to/calibration_data.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from edgewaste.oci import (
    compute_oci,
    fit_oci_weights,
    normalize_gas,
    normalize_moisture,
    run_ablation,
    select_threshold,
)
from edgewaste.oci.synthetic import DEFAULT_ANCHORS, generate_synthetic_calibration_data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit and validate the OCI model.")
    ap.add_argument("--csv", default=None, help="Real calibration_data.csv (default: synthetic).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if args.csv:
        df = pd.read_csv(args.csv)
        print(f"Loaded {len(df)} real calibration samples from {args.csv}")
    else:
        df = generate_synthetic_calibration_data(seed=args.seed)
        n_categories = df["category"].nunique()
        n_severities = df["severity_g"].nunique()
        print(f"Generated {len(df)} SYNTHETIC calibration samples "
              f"({n_categories} categories x {n_severities} severities x "
              f"{len(df) // (n_categories * n_severities)} replicates)")

    f_m = df["moisture_raw"].apply(lambda v: normalize_moisture(v, DEFAULT_ANCHORS)).to_numpy()
    f_g = df["rs"].apply(lambda v: normalize_gas(v, DEFAULT_ANCHORS)).to_numpy()
    y = df["contaminated"].to_numpy()

    print(f"\nPositive rate: {y.mean():.1%}  ({y.sum()}/{len(y)} contaminated)")

    model, diagnostics = fit_oci_weights(f_m, f_g, y)
    print("\n=== Fit diagnostics ===")
    print(json.dumps(diagnostics, indent=2))

    print("\n=== Ablation (5-fold CV AUC) ===")
    ablation = run_ablation(f_m, f_g, y)
    print(json.dumps(ablation, indent=2))

    combined_scores = np.array([
        compute_oci(model, f_m=fm, f_g=fg) for fm, fg in zip(f_m, f_g)
    ])
    threshold = select_threshold(y, combined_scores, min_sensitivity=0.95)
    print("\n=== Threshold (>=95% sensitivity, min FPR) ===")
    print(json.dumps(threshold, indent=2))

    print("\n=== OCI under all three sensor-availability cases (first 3 samples) ===")
    for i in range(min(3, len(df))):
        both = compute_oci(model, f_m=f_m[i], f_g=f_g[i])
        moisture_only = compute_oci(model, f_m=f_m[i])
        gas_only = compute_oci(model, f_g=f_g[i])
        print(f"  sample {i} (severity={df['severity_g'].iloc[i]}g, "
              f"category={df['category'].iloc[i]}): "
              f"both={both:.3f}  moisture_only={moisture_only:.3f}  gas_only={gas_only:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
