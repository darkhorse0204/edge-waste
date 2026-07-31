# PROJECT_STATUS.md
## Edge-Deployed Waste Classification with Sensor-Augmented Vision

*Owner: Ansh | Last updated: 2026-07-30*

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

### Extensions (Stage 3 — after MVP validated)
- [ ] Grad-CAM explainability (classifier)
- [ ] MC-Dropout uncertainty estimation
- [ ] Category-conditional OCI weights
- [ ] Flower federated learning
- [ ] Full ablation suite + paper draft

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

| Component | Status |
|---|---|
| Python version | ✅ 3.12.6 (system Python, no venv) |
| `edgewaste` importable | ✅ Yes (`pip install -e .` was already run) |
| `torch` | ✅ 2.8.0+cpu |
| `timm` | ✅ 1.0.25 |
| CUDA GPU | ❌ Not available — CPU only |
| `ultralytics` (YOLO26) | ❌ Not installed — run `pip install -e ".[detect]"` |
| `kaggle` CLI module | ✅ Installed |
| Kaggle credentials | ✅ `~/.kaggle/kaggle.json` exists |
| Free disk space | ⚠️ ~20 GB — enough for data + 2 checkpoints (~200MB each), but monitor it |

---

## 📅 Session Log

| Date | What was done |
|---|---|
| 2026-07-30 | Synced repo (19 files changed, new detect/ module + pipeline.py). Read HANDOVER.md + Master-Work-Plan.md. Created PROJECT_STATUS.md and implementation_plan.md. Confirmed env: Python 3.12.6, torch 2.8.0+cpu, timm 1.0.25, edgewaste importable, Kaggle creds present, ultralytics missing, ~20GB disk free. |
| 2026-07-31 | Set up Colab T4 GPU environment. Handled corrupted data in Trashbox. Successfully trained Hybrid ConvNeXt+ViT classifier — hit **95.1% val accuracy** on epoch 15! |
| 2026-07-31 | Trained YOLO26n localizer on TACO for 60 epochs overnight. Hit **0.687 mAP50**. Stage 1 training is fully complete. |
