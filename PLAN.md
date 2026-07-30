# Project Plan — Edge-Deployed Waste Classification with Sensor-Augmented Vision using Federated Learning

Source: `flowwork.docx` (project blueprint discussion). This file indexes the plan; the
detailed, actionable breakdown for each stage lives in its own file under `docs/`.

## Problem Statement

Municipal waste segregation today mostly relies on camera-only systems that:
- fail on contaminated waste
- cannot detect organic contamination
- perform poorly in real (non-lab) environments
- require constant cloud connectivity
- do not continuously learn
- cannot explain their decisions

The blueprint proposes combining AI + IoT + Edge Computing + Federated Learning +
Explainable AI into one system to address these gaps.

## End Goal

A smart waste segregation system that detects, classifies, checks contamination,
estimates its own confidence, sorts automatically, learns continuously across
deployments, and preserves data privacy (no raw data leaves the device).

## System Architecture (from blueprint)

```
Waste Item → Camera + Sensors (Camera, Moisture, Gas, Metal, Load Cell)
          → Sensor Fusion Module
          → Vision AI Model (Transformer + ConvNeXt)
          → Cross-Model Fusion
          → Waste Class Prediction ──► Organic Contamination Index
                                    ──► Bayesian Confidence
          → Decision Engine
          → Servo Motors / Conveyor Belt → Waste Bin
          → Federated Learning Client → Central FL Server
```

## The 10 Modules (as defined in the blueprint)

1. Dataset Collection (TACO, TrashNet, Garbage Classification, own images)
2. Data Preprocessing (resize, normalize, augment, split)
3. Vision Model (ConvNeXt + Vision Transformer + Attention → feature vector)
4. Sensor Module (camera, moisture, gas, metal, weight)
5. Sensor Fusion (image feature + gas + moisture + weight + metal → fusion layer)
6. Cross-Model Fusion (ConvNeXt + ViT fusion, or YOLO + ConvNeXt)
7. Organic Contamination Index (OCI) — novel, not found in existing papers
8. Confidence Estimation (Bayesian NN / MC Dropout / Temperature Scaling)
9. Decision Engine (waste class + OCI + confidence → action)
10. Federated Learning (per-bin training → FedAvg → global model, via Flower)

Supporting concerns that cut across modules: hardware selection, software stack,
the training pipeline, edge deployment flow, the conveyor-belt control loop, the
12 waste classes, the OCI formula, evaluation metrics, explainability (SHAP/LIME),
and the 6-phase academic timeline — all captured in the stage docs below where
they become relevant.

## Staging (this is the blueprint's own recommendation, not an assumption)

The blueprint explicitly recommends **not** building all 10 modules at once, but
in three stages so there's always a working system:

| Stage | Theme | Detail |
|---|---|---|
| [Stage 1](docs/stage-1-core-mvp.md) | Core MVP | Camera + dataset + Transformer/ConvNeXt model + basic waste classification |
| [Stage 2](docs/stage-2-enhanced-system.md) | Enhanced System | Sensor fusion, OCI, confidence estimation, conveyor belt control |
| [Stage 3](docs/stage-3-research-features.md) | Research Features | Federated Learning, SHAP/LIME explainability, optimization, full evaluation |

## Decisions Made

- **Taxonomy (resolved 2026-07-20, supersedes the 2026-07-18 "merge both"
  decision below): narrow to 7 classes, public data only.** Following the
  external planning docs (`Waste-Classification-Tech-Stack-Recommendation.md`,
  `Project-Action-Plan.md`), the target taxonomy is `cardboard, paper, plastic,
  glass, metal, organic, other` — sourced from TrashNet, TrashBox, and
  Garbage-Classification-12 (all Kaggle). The custom 18-class medical/PPE
  dataset is **out of scope** and no longer referenced. The single source of
  truth is [`src/edgewaste/taxonomy.py`](src/edgewaste/taxonomy.py).
  <details><summary>Superseded 2026-07-18 decision (kept for history)</summary>
  The custom dataset's 14 classes and the blueprint's 12-class list were
  combined into an 18-class taxonomy, with `hazardous` left as a known gap.
  Abandoned because it conflicted with the external planning docs' scope, and
  because the custom dataset turned out to be a Roboflow-aggregated set, not
  captured original images (see the "Custom dataset audit" this superseded in
  `docs/stage-1-core-mvp.md`).
  </details>
- **Detect-then-classify pipeline (resolved 2026-07-20), resolving the
  ConvNeXt+ViT-fusion-vs-YOLO+ConvNeXt open question below: both, staged.**
  YOLO26 (single-class, trained on TACO) localizes/crops items on a conveyor;
  the existing ConvNeXt+ViT hybrid classifies each crop. See
  [`src/edgewaste/detect/`](src/edgewaste/detect/) and
  [`src/edgewaste/pipeline.py`](src/edgewaste/pipeline.py).
- **OCI formula: `OCI-Formula-Design-and-Calibration-Protocol.md` (repo root)
  is now authoritative**, superseding the `0.4×Moisture + 0.4×Gas +
  0.2×VisionScore` draft formula still referenced in
  `docs/stage-2-enhanced-system.md`. The calibrated design is a 2-sensor
  (moisture + gas only, no vision term) logistic-regression-derived sigmoid,
  fit once offline and hard-coded for edge inference — not yet implemented,
  this only updates which spec Stage 2 should build against.
- **Core Stage-2 sensor MVP: moisture + MQ-135 only**, per
  `Project-Action-Plan.md`'s explicit deferral of the metal sensor and load
  cell to the enhanced-prototype phase. This conflicts with
  `docs/stage-2-enhanced-system.md`'s Module 4 checklist, which lists all four
  sensors as needed "for this stage" — that checklist is stale until Stage 2
  work actually starts; treat the leaner two-sensor MVP as current guidance.

## Open Questions / Things Not Yet Decided

These are called out rather than assumed, per the blueprint being a discussion
document rather than a finalized spec:

- Final hardware choice: Jetson Nano vs Raspberry Pi 5 (affects inference
  framework — TensorRT vs generic ONNX/TFLite runtime). Both external docs
  recommend deferring the Jetson purchase until Pi 5/laptop throughput is
  confirmed insufficient.
- Whether Federated Learning is simulated (single machine, multiple virtual
  clients) or deployed across genuinely separate physical bins — changes
  Stage 3 scope significantly.

## Timeline (from blueprint, academic phases — distinct from the 3 build stages above)

1. Literature survey, problem definition, objectives, architecture
2. Dataset collection/preprocessing, sensor selection, hardware procurement
3. Train vision model, build sensor fusion, implement OCI
4. Integrate sensors on edge device, add confidence estimation, connect conveyor belt
5. Federated Learning (simulate or real), performance evaluation, optimization
6. Real-world testing, documentation, final report/paper/presentation

Note the academic phases and the build stages don't map 1:1 — Phase 3 already
pulls in OCI (a Stage 2 item) while vision-model training (Stage 1) also starts
there. The stage docs are the source of truth for build order; phases are the
reporting/timeline view for the project write-up.
