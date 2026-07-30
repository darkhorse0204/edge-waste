# Stage 2 — Enhanced System

> Goal: turn the Stage 1 camera-only classifier into the full multi-modal,
> contamination-aware, physically-sorting system described in the architecture
> diagram — still on a single device, no federated learning yet.

## Scope (per blueprint)

"Add sensor fusion, OCI, confidence estimation, and conveyor belt control."

Maps to blueprint **Module 4**, **Module 5**, **Module 6**, **Module 7**,
**Module 8**, **Module 9**, plus **Edge Deployment** (section 8) and
**Conveyor Belt Logic** (section 11).

> **Scope update (2026-07-20, see PLAN.md "Decisions Made"):** the core Stage 2
> MVP is **moisture + MQ-135 only**, per `Project-Action-Plan.md`'s explicit
> phasing — metal sensor and load cell move to the enhanced-prototype tail of
> Stage 2 (or later), not the initial build. The OCI formula in Module 4 below
> is also stale: `OCI-Formula-Design-and-Calibration-Protocol.md` (repo root)
> is now the authoritative spec — 2 sensors (moisture + gas), no vision term,
> logistic-regression-fit sigmoid, not the `0.4/0.4/0.2` draft below. The rest
> of this doc (sensor fusion, cross-model fusion, confidence, decision engine,
> conveyor logic) is unaffected by that change and still applies.

## What needs to be done

### 1. Sensor Module (Module 4)
- [ ] Integrate moisture sensor (capacitive) — core MVP, do first.
- [ ] Integrate gas sensor (MQ135) — core MVP, do first. Run the mandatory
      24–48h burn-in and compute per-unit R0 before any calibration data
      collection (see the OCI calibration protocol doc, Part 1.2 and 4.2).
- [ ] Integrate DHT22 (temperature/humidity) alongside both — confound logging
      per the calibration protocol, not itself an OCI input.
- [ ] *Deferred to enhanced prototype:* metal sensor (inductive), weight
      sensor (HX711 + load cell) — see `Project-Action-Plan.md` §6 for why.
- [ ] Each sensor needs a read pipeline producing the units shown in the
      blueprint's example: moisture (%), gas (ppm), metal (boolean), weight (g).

### 2. Sensor Fusion (Module 5) — "biggest novelty" per blueprint
- [ ] Combine: image feature (from Stage 1 model) + gas + moisture + weight + metal.
- [ ] Design and implement the fusion layer producing a "final feature."
- [ ] Fusion layer architecture is unspecified in the blueprint — needs its own
      design decision (concatenation + MLP is the simplest baseline, but not
      confirmed as the intended approach).

### 3. Cross-Model Fusion (Module 6)
- [ ] Guide specifically suggested this — implement fusion between ConvNeXt and
      Vision Transformer outputs → classifier.
- [ ] Blueprint also lists "YOLO + ConvNeXt" as an alternative — this is an
      open decision (see PLAN.md), not something to pick unilaterally.

### 4. Organic Contamination Index — OCI (Module 7)
- [ ] Implement the calibrated formula from
      `OCI-Formula-Design-and-Calibration-Protocol.md` (repo root), **not**
      the `0.4×Moisture + 0.4×Gas + 0.2×VisionScore` draft this checklist item
      originally named:
      `OCI = σ(β0 + β1·f_m(moisture) + β2·f_g(gas) [+ β3·interaction])`,
      with `f_m`/`f_g` fixed-reference-normalized (not raw min-max) and `g(G)
      = -log(Rs/R0)` for the gas channel (not a ppm datasheet curve-fit).
- [ ] Fit `β0..β3` via logistic regression on staged calibration data (protocol
      doc Part 3–4: ~160 samples, 2 categories × 5 severities × 8 replicates),
      then hard-code the coefficients — no live regression at runtime.
- [ ] Select the operating threshold via sensitivity-constrained ROC (protocol
      doc Part 6), not a fixed 0–20/20–50/50–100 banding guessed up front.
- [ ] This is called out as not present in existing literature — worth
      documenting carefully for the eventual paper/report (protocol doc Part 7
      has a ready-made novelty framing).

### 5. Confidence Estimation (Module 8)
- [ ] Choose one of: Bayesian Neural Network, Monte Carlo Dropout, Temperature
      Scaling (blueprint lists all three as "possible" — pick one to implement,
      MC Dropout is typically the lowest-effort starting point but this is a
      call to make deliberately, not by default).
- [ ] Output a confidence score alongside the class prediction (e.g. "Plastic, 98%").
- [ ] Low-confidence predictions should route to manual inspection (ties into
      Decision Engine below).

### 6. Decision Engine (Module 9)
- [ ] Inputs: waste class, OCI, confidence.
- [ ] Implement the decision rules from the blueprint's examples:
  - Plastic + OCI Low + Confidence High → Recycle
  - Plastic + OCI High → Wash Required
  - Confidence Low → Manual Verification
- [ ] These three examples are illustrative, not an exhaustive rule table —
      the full decision matrix across all 12 classes × OCI bands × confidence
      levels still needs to be defined.

### 7. Conveyor Belt Logic (Module 11) + Edge Deployment (section 8)
- [ ] Loop: camera detects object → stop belt → capture image → read sensors →
      AI prediction → rotate servo → restart belt.
- [ ] Hardware: servo/stepper motor, conveyor belt mechanism.
- [ ] This is the physical control loop wrapping everything built so far —
      requires the edge device to run inference, sensor fusion, and decision
      logic all within the belt-stop window.

## Hardware needed for this stage (in addition to Stage 1)
- Core MVP: MQ135 (gas), capacitive moisture sensor, DHT22, ESP32/Arduino
  (+ ADS1115 if reading analog sensors from a Raspberry Pi).
- Deferred to enhanced prototype: inductive metal sensor, HX711 + load cell,
  servo/stepper motor + conveyor belt.

## Software needed for this stage (in addition to Stage 1)
- FastAPI (if sensor/decision services are split into separate processes/services).
- Firebase (blueprint lists this — likely for logging/telemetry, not confirmed).

## Exit criteria for Stage 2

- End-to-end physical demo: an item placed on the belt is detected, sensed,
  classified, scored for contamination and confidence, routed by the decision
  engine, and physically sorted by the servo.
- OCI and confidence values logged per item for later evaluation (Stage 3 needs
  this data for the full metrics suite).

## Explicitly NOT in Stage 2
- Federated Learning across multiple bins — Stage 3.
- SHAP/LIME explainability — Stage 3.
- Model optimization (quantization/pruning) and full-scale evaluation — Stage 3.
