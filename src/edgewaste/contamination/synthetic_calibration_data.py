# synthetic_calibration_data.py - synthetic sensor calibration data, used until real sensors exist
"""Synthetic calibration data for testing the OCI fitting pipeline before
real sensor hardware exists. Mirrors the real protocol's design (Part 4):
2 categories x 5 severity levels x N replicates. Delete once real
calibration data replaces it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgewaste.contamination.sensor_normalization import CalibrationAnchors


def generate_synthetic_calibration_data(
    n_replicates: int = 8, seed: int = 42, noise_scale: float = 1.0,
    anchors: CalibrationAnchors | None = None,
) -> tuple[pd.DataFrame, CalibrationAnchors]:
    """noise_scale multiplies every noise term (1.0 = the default sensors); pass
    `anchors` to reuse the calibration constants of an earlier draw, so a
    separate test set is normalised exactly as a deployed unit would be."""
    rng = np.random.default_rng(seed)
    categories = ["porous", "rigid"]
    severities_g = [0, 2, 5, 10, 20]

    rows = []
    for category in categories:
        damping = 0.4 if category == "rigid" else 1.0
        for severity_g in severities_g:
            for _ in range(n_replicates):
                # two mostly-independent latent cues -- moisture picks up
                # surface wetness, gas picks up decomposition/voc off-gassing;
                # real items don't show both equally (dry-but-smelly vs.
                # wet-but-odorless), so these shouldn't be near-perfectly
                # correlated.
                wetness_cue = damping * severity_g + rng.normal(0, 6.0 * noise_scale)
                decomp_cue = damping * severity_g + rng.normal(0, 6.0 * noise_scale)

                moisture_raw = 1500 + wetness_cue * 40 + rng.normal(0, 80 * noise_scale)
                rs_over_r0 = max(0.05, 1.0 - decomp_cue * 0.03 + rng.normal(0, 0.05 * noise_scale))

                rows.append({
                    "category": category,
                    "residue_mass_g": severity_g,
                    "moisture_raw": moisture_raw,
                    "rs_over_r0": rs_over_r0,
                    "contaminated": int(severity_g >= 5),
                })

    df = pd.DataFrame(rows)
    if anchors is not None:
        return df, anchors
    anchors = CalibrationAnchors(
        m_dry=float(df["moisture_raw"].min()) - 50,
        m_wet=float(df["moisture_raw"].max()) + 50,
        r0=1.0,
        l_min=-np.log(df["rs_over_r0"].max()),
        l_max=-np.log(df["rs_over_r0"].min()),
    )
    return df, anchors