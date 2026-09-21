# PROJECT_STATUS.md
## Edge-Deployed Waste Classification with Sensor-Augmented Vision

*Owner: Ansh | Last updated: 2026-09-22*

**2026-09-22 note:** this file was stale for 7 weeks (still said CPU-only,
no ultralytics). A separate audit (`EdgeWaste_Project_Report.pdf`, 3 Aug)
described a much more advanced state — trained checkpoints, OCI, Grad-CAM,
12,751 logged detections — but that work lived only on a different machine
and was never committed; this working directory had none of it (empty
`runs/`, only the original 3 commits). Tonight's session (ahead of the
Review-1 deadline) rebuilt Stage 2/3 in software with hardware simulated —
see README.md's "What's built (Stage 2/3, hardware simulated)" table — and
kicked off classifier + detector training on Colab (this machine's GPU is
only 4GB; Colab's T4 is what actually produced the 94.95%/0.687 mAP50
numbers the Aug 3 audit reported). Update the training-result rows below
once Colab finishes.

---

## 🗂️ Document Map

| File | Role |
|---|---|
| **`PROJECT_STATUS.md`** (this) | Living status tracker — updated every session |
| `HANDOVER.md` | Full "what exists + what to do next" narrative — read first |
| `Master-Work-Plan.md` | Granular execution roadmap with code snippets and pin diagrams |
| `PLAN.md` | Decision log and module map |
| `README.md` | Command reference |

---

## 📍 Current Phase

**PHASE 1 → PHASE 2 Transition: Environment ready (code), data pipeline is next.**

The entire vision pipeline (code only) is complete. No data has been downloaded,
no model has been trained. The immediate blocker is Kaggle credentials.

---

## ✅ What Is DONE (Code Complete, Untested)

### Vision Pipeline — Stage 1 (Classifier)
- [x] `taxonomy.py` — 7-class taxonomy (cardboard, paper, plastic, glass, metal, organic, other) + 3 source mappings
- [x] `config.py` — YAML-driven typed config (DataConfig, ModelConfig, TrainConfig, DetectConfig)
- [x] `data/kaggle_sources.py` — `edgewaste-fetch`: downloads TrashNet, TrashBox, Garbage-12 from Kaggle
- [x] `data/ingest.py` — `edgewaste-ingest`: consolidates downloads → `data/processed/<class>/`
- [x] `data/splits.py` — `edgewaste-split`: stratified 70/15/15 split → `data/splits.csv`
- [x] `data/dataset.py` — `WasteDataset`: augmentations, class weights, sampler weights
- [x] `models/hybrid.py` — `HybridConvNeXtViT`: ConvNeXt-Tiny + ViT-Small with attention fusion
- [x] `train.py` — `edgewaste-train`: full training loop (AMP, weighted sampler, early stopping, OneCycleLR)
- [x] `evaluate.py` — `edgewaste-eval`: confusion matrix + per-class metrics
- [x] `infer.py` — `edgewaste-infer`: single image or webcam inference

### Vision Pipeline — Detector (YOLO26)
- [x] `detect/data.py` — `edgewaste-detect-fetch`: downloads TACO (YOLO-format Kaggle mirror), collapses to 1-class `waste_item`
- [x] `detect/train.py` — `edgewaste-detect-train`: fine-tunes YOLO26n
- [x] `pipeline.py` — `edgewaste-pipeline`: detector → crop → classifier end-to-end

### Configuration
- [x] `configs/stage1.yaml` — classification pipeline config
- [x] `configs/detect.yaml` — detection pipeline config
- [x] `pyproject.toml` — installable package with CLI entry points
- [x] `requirements.txt` — pinned deps

---

## ⏳ What Is NOT YET STARTED

### Immediate (Vision Data + Training)
- [x] Configure Kaggle credentials (`~/.kaggle/kaggle.json`)
- [x] Download Stage 1 vision datasets (TrashNet, TrashBox, Garbage-12)
- [x] Consolidate images and build train/val/test splits (`edgewaste-ingest`, `edgewaste-split`)
- [x] Train baseline hybrid classifier (`edgewaste-train`) — **Achieved 95.1% validation accuracy!**
- [x] Evaluate baseline classifier on test split (`edgewaste-eval`)
- [x] Download TACO dataset and prepare for YOLO (`edgewaste-detect-fetch`)
- [x] Train YOLO26 single-class waste localizer (`edgewaste-detect-train`) — **Achieved 68.7% mAP50**
- [ ] Run end-to-end inference (`edgewaste-pipeline`)

### Hardware & Sensors (Stage 2 — nothing started)
- [ ] ESP32 + sensors bring-up (moisture, MQ-135, DHT11)
- [ ] MQ-135 burn-in (24-48h), R0 calibration
- [ ] Raspberry Pi 5 setup + headless SSH
- [ ] Camera test on Pi
- [ ] ESP32 → Pi USB serial verified

### OCI — Organic Contamination Index (Stage 2)
- [ ] Physical calibration data collection (160 staged samples)
- [ ] Feature engineering (`f_m`, `f_g` normalization)
- [ ] Logistic regression fit (beta0, beta1, beta2)
- [ ] Ablation study (moisture-only, gas-only, combined AUC)
- [ ] Threshold selection (sensitivity-constrained)
- [ ] `compute_oci()` with sensor-dropout fallback

### Edge Deployment (Stage 2)
- [ ] ONNX export (classifier + detector)
- [ ] On-Pi benchmark (<500ms per item target)
- [ ] INT8 quantization (if latency target not met)

### Extensions (Stage 3)
- [x] Grad-CAM explainability (classifier, ConvNeXt branch) — `explain.py`
- [x] MC-Dropout uncertainty estimation — `confidence.py`
- [ ] Category-conditional OCI weights
- [x] Federated learning — single-process FedAvg simulation (`federated/`); real Flower/physical-device deployment not done
- [ ] Full ablation suite + paper draft

### Built tonight (2026-09-22) — software demo, hardware simulated
- [x] `oci/` package: fixed-reference normalisation, 3-model fit (combined/moisture-only/gas-only), 5-fold CV ablation, sensitivity-constrained threshold — fitted on synthetic calibration data (`scripts/fit_oci_demo.py`)
- [x] `sensors.py` — simulated moisture/gas/metal/load-cell readings, per-class-plausible with injected contamination events
- [x] `decision.py` — decision engine (uncertainty -> manual review; OCI -> contaminated reject; else -> material bin) + mock conveyor actuator (logs the action)
- [x] `confidence.py` — MC-Dropout uncertainty (stochastic passes, dropout-only, BatchNorm untouched)
- [x] `explain.py` + `logging_utils.py` — Grad-CAM on keypress, per-detection CSV logging (unchanged in spirit from the Aug 3 audit's description, rewritten from scratch since that code no longer existed here)
- [x] `federated/simulate.py` — FedAvg simulation (IID + non-IID shard partitioning, sample-weighted averaging), `edgewaste-federated` CLI
- [x] `export_onnx.py` — classifier + detector ONNX export with PyTorch-parity verification, `edgewaste-export` CLI
- [x] `pipeline.py` rewritten to integrate all of the above per-frame; `run_camera.py` session launcher
- [x] All of the above unit- and integration-tested locally (stub detector + real image, tiny synthetic FL dataset, dummy-checkpoint ONNX export) — all passed
- [ ] SHAP/LIME for OCI — not done, lowest priority given the deadline
- [ ] Not done: real ONNX-on-Pi benchmark (no Pi), real hardware calibration (48h burn-in blocker, unchanged from the Aug 3 audit)

---

## 🚧 Known Risks / Open Decisions

| Risk | Status | Notes |
|---|---|---|
| Kaggle folder renames | Active risk | `ingest.py` raises a clear `KeyError` if folder names changed — fix mapping in `taxonomy.py` |
| TACO YOLO mirror layout | Unverified | `detect/data.py`'s `_find_pairs()` may need tweaks if layout differs |
| YOLO26 `ultralytics` version | Pinned `>=8.4` floor | Check `docs.ultralytics.com/models/yolo26` if `YOLO("yolo26n.pt")` fails |
| Disk space | ⚠️ Critical | Machine was at 99% during development — each checkpoint ~200MB |
| 1-class vs 7-class detector | Open decision | Current: 1-class `waste_item`. MWP proposes 7-class. Decide before training. |
| ZeroWaste-f | Not wired | No confirmed Kaggle mirror; requires manual download from ai.bu.edu |

---

## 📊 Training Targets (Not Yet Measured)

| Metric | Target | Actual |
|---|---|---|
| Classifier test accuracy | ≥ 80% | — |
| Detector mAP50 | Record + report | — |
| OCI sensitivity (recall on contaminated) | ≥ 90–95% | — |
| End-to-end Pi latency | < 500ms/item | — |

---

## 🔧 Environment Status

| Component | Status (2026-09-22) |
|---|---|
| Python version | 3.11 (Windows Store install, local-packages) |
| `edgewaste` importable | ✅ Yes |
| `torch` | ✅ 2.5.1+cu121 |
| `timm` | ✅ 1.0.29 |
| CUDA GPU | ✅ NVIDIA RTX 2050, **4GB VRAM** — fine for oci/confidence/decision/federated-smoke work, too small for full-dataset 50.7M-param training in reasonable time; classifier/detector training done on Colab's T4 instead |
| `ultralytics` (YOLO26) | not yet installed locally (only needed for `edgewaste-pipeline`/`run_camera.py` once checkpoints exist) |
| `grad-cam`, `onnx`, `onnxruntime`, `opencv-python`, `torchvision` | ✅ Installed tonight |
| `kaggle` CLI module | not verified tonight — Colab notebook sets up its own credentials from Drive |
| Free disk space | ⚠️ 13 GB free on C: (98% full) — same risk flagged before; checkpoint writes are atomic |

---

## 📅 Session Log

| Date | What was done |
|---|---|
| 2026-07-30 | Synced repo (19 files changed, new detect/ module + pipeline.py). Read HANDOVER.md + Master-Work-Plan.md. Created PROJECT_STATUS.md and implementation_plan.md. Confirmed env: Python 3.12.6, torch 2.8.0+cpu, timm 1.0.25, edgewaste importable, Kaggle creds present, ultralytics missing, ~20GB disk free. |
| 2026-07-31 | Set up Colab T4 GPU environment. Handled corrupted data in Trashbox. Successfully trained Hybrid ConvNeXt+ViT classifier — hit **95.1% val accuracy** on epoch 15! |
| 2026-07-31 | Trained YOLO26n localizer on TACO for 60 epochs overnight. Hit **0.687 mAP50**. Stage 1 training is fully complete. |
| 2026-09-22 | Found this working directory had regressed to "code complete, untrained" (empty `runs/`, only 3 original commits) despite an Aug 3 audit describing trained checkpoints + OCI + live sessions — that work only ever existed on a different machine and was never committed. Read all 4 faculty documents (Zero Review approval, Review-1 report, the Aug 3 audit). User needs the project "complete" by tonight for review; scoped to a full software demo with physical hardware simulated (agreed via clarifying questions). Built and locally tested: `oci/` package, `sensors.py`, `decision.py`, `confidence.py` (MC-Dropout), `explain.py` (Grad-CAM), `logging_utils.py`, `federated/simulate.py` (FedAvg simulation), `export_onnx.py`, rewrote `pipeline.py` to integrate everything, added `run_camera.py`. Kicked off classifier + detector training on Colab (`colab_training.ipynb`) since this machine's 4GB GPU can't train the 50.7M-param model in reasonable time. |
