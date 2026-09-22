"""Simulated physical sensors — stand-in for the ESP32 + moisture + MQ-135 +
DHT11 + metal-detector + load-cell hardware that hasn't been built yet.

No physical sensor has ever been wired in this project (see the project
audit's Stage-2 gap register), so this module exists purely so the full
"camera -> classify -> sensor fusion -> OCI -> decision -> conveyor" pipeline
can be demonstrated end-to-end in software. Readings are drawn from
distributions that are *plausible per predicted material* (an "organic"
crop tends to read wetter/higher-gas than a "glass" crop) with injected
noise and occasional contamination events, so the OCI stage has something
non-trivial to react to. Swap `simulate_sensors` for a real serial read from
the ESP32 the moment hardware exists — nothing downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .oci.normalize import CalibrationAnchors

# Same anchors as oci/synthetic.py so simulated readings and the OCI model
# fitted on synthetic calibration data speak the same units.
SIM_ANCHORS = CalibrationAnchors(m_dry=200.0, m_wet=800.0, r0=10_000.0, l_min=0.0, l_max=2.5)

# Per-class base contamination propensity (0-1): how "wet/off-gassing" a
# freshly-classified item of this material typically reads, before noise.
# Organic is deliberately wet/high-gas by construction; everything else is
# low but non-zero, so an occasional "greasy pizza box" (contaminated
# cardboard) is possible and gives OCI something to catch.
_BASE_WETNESS = {
    "organic": 0.85, "cardboard": 0.15, "paper": 0.15, "plastic": 0.10,
    "glass": 0.05, "metal": 0.05, "textile": 0.20, "battery": 0.05,
    "e_waste": 0.05, "medical": 0.30, "other": 0.20,
}
_METAL_TRUE_RATE = {"metal": 0.95, "battery": 0.80, "e_waste": 0.70, "other": 0.05}


@dataclass
class SensorReading:
    moisture_raw: float
    rs: float  # MQ-135 sensor resistance
    metal_detected: bool
    load_g: float


def simulate_sensors(
    class_name: str, rng: np.random.Generator | None = None,
    contamination_event_prob: float = 0.08,
) -> SensorReading:
    rng = rng or np.random.default_rng()
    base = _BASE_WETNESS.get(class_name, 0.15)
    if rng.random() < contamination_event_prob and class_name != "organic":
        base = min(1.0, base + rng.uniform(0.4, 0.7))  # a contamination event

    wetness = float(np.clip(base + rng.normal(0, 0.08), 0, 1))
    offgas = float(np.clip(base + rng.normal(0, 0.10), 0, 1))

    moisture_raw = SIM_ANCHORS.m_dry + wetness * (SIM_ANCHORS.m_wet - SIM_ANCHORS.m_dry)
    rs = SIM_ANCHORS.r0 * np.exp(-(SIM_ANCHORS.l_min + offgas * (SIM_ANCHORS.l_max - SIM_ANCHORS.l_min)))

    metal_rate = _METAL_TRUE_RATE.get(class_name, 0.03)
    metal_detected = bool(rng.random() < metal_rate)
    load_g = float(max(1.0, rng.normal(60, 40)))

    return SensorReading(moisture_raw=moisture_raw, rs=float(rs),
                          metal_detected=metal_detected, load_g=load_g)
