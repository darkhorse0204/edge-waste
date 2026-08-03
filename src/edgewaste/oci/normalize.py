"""Fixed-reference normalization for the moisture and gas channels.

Per the OCI protocol doc, Part 1: NOT running min-max over live data (fragile
to a single outlier redefining the scale) and NOT a datasheet ppm curve-fit
for gas (extra approximation error). Both anchors are measured once, offline,
during calibration, then hard-coded as constants for deployment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationAnchors:
    """One-time calibration constants for a specific physical sensor pair.
    Measure these per the OCI protocol doc Part 4.2 (fixed dry/wet, clean-air/
    saturated reference readings) — do not recompute at runtime."""
    m_dry: float    # moisture_raw on a genuinely dry reference item
    m_wet: float    # moisture_raw on a genuinely saturated reference item
    r0: float       # MQ-135 clean-air baseline resistance (per-unit, mandatory)
    l_min: float    # -log(Rs/R0) in clean air
    l_max: float    # -log(Rs/R0) near a heavily contaminated reference


def normalize_moisture(moisture_raw: float, anchors: CalibrationAnchors) -> float:
    span = anchors.m_wet - anchors.m_dry
    if span == 0:
        raise ValueError("m_wet == m_dry — calibration anchors invalid.")
    f_m = (moisture_raw - anchors.m_dry) / span
    return max(0.0, min(1.0, f_m))


def normalize_gas(rs: float, anchors: CalibrationAnchors) -> float:
    """rs: sensor resistance computed from the raw ADC reading, per the
    R0-calibration formula in Master-Work-Plan.md Sub-Phase 1.2, step 8."""
    if rs <= 0 or anchors.r0 <= 0:
        raise ValueError("Rs and R0 must be positive — check sensor wiring/calibration.")
    g_signal = -math.log(rs / anchors.r0)
    span = anchors.l_max - anchors.l_min
    if span == 0:
        raise ValueError("l_max == l_min — calibration anchors invalid.")
    f_g = (g_signal - anchors.l_min) / span
    return max(0.0, min(1.0, f_g))