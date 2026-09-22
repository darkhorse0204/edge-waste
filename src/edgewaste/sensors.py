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
from .taxonomy import CLASS_TO_FAMILY

# Same anchors as oci/synthetic.py so simulated readings and the OCI model
# fitted on synthetic calibration data speak the same units.
SIM_ANCHORS = CalibrationAnchors(m_dry=200.0, m_wet=800.0, r0=10_000.0, l_min=0.0, l_max=2.5)

# Per-class base contamination propensity (0-1): how "wet/off-gassing" a
# freshly-classified item of this material typically reads, before noise.
# Organic is deliberately wet/high-gas by construction; everything else is
# low but non-zero, so an occasional "greasy pizza box" (contaminated
# cardboard) is possible and gives OCI something to catch.
# Keyed by material *family*: wetness is a property of what the item is made
# of and what it held, not of which specific bottle variant it is.
_BASE_WETNESS = {
    "organic": 0.85, "cardboard": 0.15, "paper": 0.15, "plastic": 0.10,
    "glass": 0.05, "metal": 0.05, "styrofoam": 0.12, "textile": 0.20,
    "hazardous": 0.10,
}
_METAL_TRUE_RATE = {"metal": 0.95, "hazardous": 0.60}


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
    """`class_name` is a fine-grained item class; readings are drawn from its
    material family's profile."""
    rng = rng or np.random.default_rng()
    family = CLASS_TO_FAMILY.get(class_name, class_name)
    base = _BASE_WETNESS.get(family, 0.15)
    if rng.random() < contamination_event_prob and family != "organic":
        base = min(1.0, base + rng.uniform(0.4, 0.7))  # a contamination event

    wetness = float(np.clip(base + rng.normal(0, 0.08), 0, 1))
    offgas = float(np.clip(base + rng.normal(0, 0.10), 0, 1))

    moisture_raw = SIM_ANCHORS.m_dry + wetness * (SIM_ANCHORS.m_wet - SIM_ANCHORS.m_dry)
    rs = SIM_ANCHORS.r0 * np.exp(-(SIM_ANCHORS.l_min + offgas * (SIM_ANCHORS.l_max - SIM_ANCHORS.l_min)))

    metal_rate = _METAL_TRUE_RATE.get(family, 0.03)
    metal_detected = bool(rng.random() < metal_rate)
    load_g = float(max(1.0, rng.normal(60, 40)))

    return SensorReading(moisture_raw=moisture_raw, rs=float(rs),
                          metal_detected=metal_detected, load_g=load_g)


def fit_demo_oci_model():
    """Fit an OCI model on synthetic calibration data, for demo use.

    The fitted model consumes f_m/f_g already normalised to [0, 1], so the
    calibration data and the live readings may sit on different raw scales
    as long as each is normalised with its own matching anchors — the
    calibration set uses the anchors its generator returns, live readings
    use SIM_ANCHORS above. Replace the generator with a real
    calibration_data.csv load and nothing else changes.
    """
    from .oci import fit_oci_weights, normalize_gas, normalize_moisture
    from .oci.synthetic import generate_synthetic_calibration_data

    df, anchors = generate_synthetic_calibration_data()
    f_m = df["moisture_raw"].apply(lambda v: normalize_moisture(v, anchors)).to_numpy()
    f_g = df["rs_over_r0"].apply(lambda v: normalize_gas(v, anchors)).to_numpy()
    y = df["contaminated"].to_numpy()
    model, _ = fit_oci_weights(f_m, f_g, y)
    return model


def reading_to_features(reading: SensorReading) -> tuple[float, float]:
    """Normalise a simulated reading into the (f_m, f_g) the OCI model takes."""
    from .oci import normalize_gas, normalize_moisture

    return (normalize_moisture(reading.moisture_raw, SIM_ANCHORS),
            normalize_gas(reading.rs, SIM_ANCHORS))
