# Stage 3 — Research Features

> Goal: the pieces that distinguish this from a normal capstone project and make
> it publishable — cross-device learning, explainability, and rigorous evaluation.

## Scope (per blueprint)

"Add Federated Learning, SHAP/LIME explainability, optimization, and large-scale evaluation."

Maps to blueprint **Module 10**, **Explainability** (section 10), and
**Evaluation Metrics** (section 14).

## What needs to be done

### 1. Federated Learning (Module 10)
- [ ] Decide: simulated FL (one machine, multiple virtual clients) vs. real FL
      across multiple physical bins. This changes infrastructure needs
      significantly — flagged as an open question in PLAN.md, not decided yet.
- [ ] Each client (bin) trains locally on its own data.
- [ ] Clients send model weights (not raw data) to a central server.
- [ ] Server runs FedAvg to produce a global model.
- [ ] Global model is returned to all clients ("Update Everyone").
- [ ] Framework: Flower (specified in blueprint).

### 2. Explainability (section 10)
- [ ] Guide specifically suggested SHAP and LIME — implement both, not just one,
      per the blueprint's explicit mention of both.
- [ ] Goal: explain *why* the model predicted a given class (e.g. heatmap showing
      bottle cap / bottle body / shape contributing to a "Plastic" prediction).
- [ ] Decide what "explain" outputs to — a diagnostic tool for developers, a
      user-facing display, or both (not specified in blueprint).

### 3. Optimization
- [ ] Quantization and/or pruning "if needed" (blueprint phrasing — this is
      conditional on the model being too large/slow for the target edge device,
      not an unconditional requirement).
- [ ] Should be informed by the Model Size / Memory Usage / Inference Time /
      FPS metrics gathered below — optimize based on measured bottlenecks,
      not preemptively.

### 4. Full Evaluation (section 14)
- [ ] Accuracy, Precision, Recall, F1-score, mAP.
- [ ] Inference Time, FPS.
- [ ] Energy Consumption.
- [ ] Model Size, Memory Usage.
- [ ] Communication Cost (FL-specific — bandwidth/rounds needed for FedAvg convergence).
- [ ] This is the point where Stage 1/2 logged data (predictions, OCI, confidence)
      gets aggregated into the report-ready metrics.

### 5. Final Documentation (ties to blueprint Timeline Phase 6)
- [ ] Real-world testing across scenarios (not just held-out test set).
- [ ] Final report, paper, and presentation.

## Software needed for this stage (in addition to Stages 1–2)
- Flower (federated learning framework).
- SHAP, LIME.
- TensorFlow — listed as "optional" in blueprint; only needed if a TF-specific
  tool (e.g. TFLite for edge export) is chosen over the PyTorch-only path.

## Exit criteria for Stage 3

- Multiple simulated or real bins demonstrably improve a shared global model
  via FedAvg without sharing raw images/sensor data.
- Every prediction can be paired with a SHAP/LIME explanation on demand.
- A complete metrics table (section 14 list) is filled in from real test runs,
  not estimated.

## Final Novel Contributions (per blueprint, achieved once all 3 stages complete)

- Transformer + ConvNeXt hybrid vision model for waste classification.
- Multi-modal sensor fusion (camera, moisture, gas, metal, weight).
- Organic Contamination Index (OCI) for contamination-aware segregation.
- Bayesian confidence estimation for reliable decision-making.
- Cross-model feature fusion for classification robustness.
- Conveyor-belt-based automated segregation.
- Explainable AI via SHAP and LIME.
- Edge AI deployment on Raspberry Pi or Jetson.
- Federated Learning for privacy-preserving continuous improvement across bins.
- A complete, scalable smart waste segregation framework for campuses/smart cities.
