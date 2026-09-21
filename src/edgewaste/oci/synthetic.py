"""Synthetic calibration data — stand-in until the real 160-sample physical
protocol (ESP32 + moisture + MQ-135, 24-48h burn-in, staged food-waste
severities) can be run.

Mirrors the real protocol's experimental design: 2 categories (porous, rigid)
x 5 severity levels (0/2/5/10/20 g of food residue) x 8 replicates = 80
samples per category pairing, rigid items damped to 0.4x to model their
enclosure blocking off-gassing/wetness transfer.

Generates two mostly-independent latent cues (surface wetness, decomposition
off-gassing) rather than deriving both sensor channels from one variable —
real items are sometimes dry-but-smelly or wet-but-odourless, and a fully
correlated generator would make the moisture-vs-gas ablation meaningless by
construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .normalize import CalibrationAnchors

DEFAULT_ANCHORS = CalibrationAnchors(m_dry=200.0, m_wet=800.0, r0=10_000.0, l_min=0.0, l_max=2.5)

CATEGORIES = ("porous", "rigid")
SEVERITIES_G = (0, 2, 5, 10, 20)  # grams of staged food residue
REPLICATES = 8
RIGID_DAMPING = 0.4
CONTAMINATION_THRESHOLD_G = 5  # >= this many grams counts as "contaminated" (label)


def generate_synthetic_calibration_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for category in CATEGORIES:
        damp = RIGID_DAMPING if category == "rigid" else 1.0
        for severity_g in SEVERITIES_G:
            for _ in range(REPLICATES):
                severity_norm = severity_g / max(SEVERITIES_G)
                # Two mostly-independent latent cues: each is only half driven
                # by the shared severity signal, half by its own independent
                # random draw (a dry-but-smelly or wet-but-odourless item), so
                # the two sensor channels are correlated with ground truth but
                # not collinear with each other.
                wetness_latent = np.clip(
                    damp * (0.5 * severity_norm + 0.5 * rng.uniform(0, 1))
                    + rng.normal(0, 0.05),
                    0, 1,
                )
                offgas_latent = np.clip(
                    damp * (0.5 * severity_norm + 0.5 * rng.uniform(0, 1))
                    + rng.normal(0, 0.05),
                    0, 1,
                )
                moisture_raw = DEFAULT_ANCHORS.m_dry + wetness_latent * (
                    DEFAULT_ANCHORS.m_wet - DEFAULT_ANCHORS.m_dry
                )
                # Higher off-gassing -> lower Rs/R0 -> larger -log(Rs/R0).
                rs = DEFAULT_ANCHORS.r0 * np.exp(
                    -(DEFAULT_ANCHORS.l_min + offgas_latent * (DEFAULT_ANCHORS.l_max - DEFAULT_ANCHORS.l_min))
                )
                rows.append(
                    {
                        "category": category,
                        "severity_g": severity_g,
                        "moisture_raw": moisture_raw,
                        "rs": rs,
                        "contaminated": int(severity_g >= CONTAMINATION_THRESHOLD_G),
                    }
                )
    return pd.DataFrame(rows)
