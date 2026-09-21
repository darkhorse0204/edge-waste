"""Fixed-reference feature scaling for the two OCI sensor channels.

Deliberately NOT min-max over live data: a single outlier reading would
silently redefine the whole scale mid-deployment. Anchors are frozen
per-unit constants from a one-time calibration (clean-air R0, dry/wet
moisture extremes), not something the running system ever updates itself.

The gas channel works directly in Rs/R0 log-space rather than fitting the
MQ-135 datasheet's ppm curve, so the model doesn't stack the datasheet's
approximation error on top of its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationAnchors:
    """Per-unit calibration constants, fixed at setup time."""

    m_dry: float  # moisture sensor raw reading in bone-dry air
    m_wet: float  # moisture sensor raw reading fully saturated
    r0: float  # MQ-135 sensor resistance in clean air, post burn-in
    l_min: float  # g_signal floor observed during calibration (clean air)
    l_max: float  # g_signal ceiling observed during calibration (max contamination)


def normalize_moisture(moisture_raw: float, anchors: CalibrationAnchors) -> float:
    """f_m = clip((moisture_raw - m_dry) / (m_wet - m_dry), 0, 1)."""
    span = anchors.m_wet - anchors.m_dry
    if span == 0:
        raise ValueError("Degenerate moisture anchors: m_wet == m_dry.")
    f_m = (moisture_raw - anchors.m_dry) / span
    return min(1.0, max(0.0, f_m))


def normalize_gas(rs: float, anchors: CalibrationAnchors) -> float:
    """f_g = clip((-log(Rs/R0) - l_min) / (l_max - l_min), 0, 1)."""
    if rs <= 0 or anchors.r0 <= 0:
        raise ValueError("Rs and R0 must be positive (sensor resistance readings).")
    g_signal = -math.log(rs / anchors.r0)
    span = anchors.l_max - anchors.l_min
    if span == 0:
        raise ValueError("Degenerate gas anchors: l_max == l_min.")
    f_g = (g_signal - anchors.l_min) / span
    return min(1.0, max(0.0, f_g))
