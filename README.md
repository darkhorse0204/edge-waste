# edge-waste

Edge-deployed waste classification with sensor-augmented vision and federated
learning. See [PLAN.md](PLAN.md) and [docs/](docs/) for the full staged blueprint,
and the three planning docs at the repo root (tech-stack recommendation, project
action plan, OCI calibration protocol) for the converged design this now follows.

**Status:** Stage 1 (detect-then-classify vision pipeline) is implemented and
wired end-to-end. Stage 2 (OCI, confidence, decision engine, conveyor
control) and Stage 3 (federated learning, explainability) are now also
implemented and wired end-to-end **in software, with physical hardware
simulated** — see "What's built (Stage 2/3, hardware simulated)" below. No
ESP32/sensors/Raspberry Pi/conveyor exist yet; OCI is fitted on synthetic
calibration data (not the real 160-sample physical protocol) and Federated
Learning runs as a single-process FedAvg simulation (not physically
distributed devices). Swapping either for real hardware/deployment is a
scoped, documented next step, not a rewrite — see the module docstrings.

## What's built (Stage 1)

**Taxonomy decision (supersedes the 2026-07-18 "merge 18-class" decision):**
narrowed to the **7-class recycling taxonomy** (paper, cardboard, plastic,
glass, metal, organic, other), sourced entirely from public datasets — see
[taxonomy.py](src/edgewaste/taxonomy.py). The custom 18-class medical/PPE
dataset is out of scope for this taxonomy and no longer wired in.

**Pipeline decision:** detect-then-classify, per the tech-stack doc §2.3 —
**YOLO26** (single-class item localizer, trained on TACO) finds and crops
items, then the existing **ConvNeXt + Vision Transformer hybrid** classifier
(attention-based feature fusion) types the crop into one of the 7 classes.

```
                              ┌─ detection dataset (TACO, YOLO-format) ─┐
                              │        edgewaste.detect.data            │
                              │               │                        │
                              │        runs/detect/.../best.pt         │
                              └──────────────────┬───────────────────--┘
                                                  │
classification sources ──ingest──► data/processed/<class>/  ──split──► data/splits.csv
                                                                            │
                                                    train ◄─────────────────┘
                                                      │
                                              runs/stage1/best.pt
                                                      │
                                    detector + classifier ──► edgewaste.pipeline
```

| Module | File |
|---|---|
| Canonical taxonomy + source mappings | [src/edgewaste/taxonomy.py](src/edgewaste/taxonomy.py) |
| Config (YAML → dataclasses) | [src/edgewaste/config.py](src/edgewaste/config.py), [configs/stage1.yaml](configs/stage1.yaml), [configs/detect.yaml](configs/detect.yaml) |
| Consolidate sources → canonical classes | [src/edgewaste/data/ingest.py](src/edgewaste/data/ingest.py) |
| Download public datasets (Kaggle) | [src/edgewaste/data/kaggle_sources.py](src/edgewaste/data/kaggle_sources.py) |
| Stratified train/val/test split | [src/edgewaste/data/splits.py](src/edgewaste/data/splits.py) |
| Dataset + augmentations | [src/edgewaste/data/dataset.py](src/edgewaste/data/dataset.py) |
| ConvNeXt+ViT hybrid model | [src/edgewaste/models/hybrid.py](src/edgewaste/models/hybrid.py) |
| Training loop | [src/edgewaste/train.py](src/edgewaste/train.py) |
| Evaluation + confusion matrix | [src/edgewaste/evaluate.py](src/edgewaste/evaluate.py) |
| Classifier-only inference demo (files + webcam) | [src/edgewaste/infer.py](src/edgewaste/infer.py) |
| TACO fetch + single-class YOLO label prep | [src/edgewaste/detect/data.py](src/edgewaste/detect/data.py) |
| YOLO26 detector training | [src/edgewaste/detect/train.py](src/edgewaste/detect/train.py) |
| Detect-then-classify pipeline (files + webcam) | [src/edgewaste/pipeline.py](src/edgewaste/pipeline.py) |

## Two-level taxonomy: object identity × material

The system predicts at two levels from two independent models, then
cross-checks them against each other:

| Level | Model | Classes |
|---|---|---|
| **What object is it?** | YOLO26n detector (TACO, 18 classes) | Aluminium foil, Bottle cap, Bottle, Broken glass, Can, Carton, Cigarette, Cup, Lid, Other litter, Other plastic, Paper, Plastic bag-wrapper, Plastic container, Pop tab, Straw, Styrofoam piece, Unlabeled litter |
| **What is it made of?** | ConvNeXt+ViT hybrid classifier | cardboard, paper, plastic, glass, metal, organic, other |
| **How contaminated is it?** | OCI (moisture + gas sensor fusion) | continuous 0–1 score |
| **How sure are we?** | MC-Dropout | normalised predictive entropy |

The **Object-Identity Prior** ([identity_prior.py](src/edgewaste/identity_prior.py))
couples the first two: a crop the detector confidently calls a `Can` has its
material distribution re-weighted toward `metal`. The prior is *soft* — a
confident contradictory reading survives and is flagged rather than
overwritten, because a system that can never report a mislabelled object is
worse than the error it prevents. Surviving contradictions route to manual
review.

## Video survey mode

`edgewaste-video` processes video footage of multi-item scenes (a beach
survey walk, a conveyor run, a drone pass) and produces a **de-duplicated
litter inventory**.

Counting — not detection — is the hard part: one physical item spans
hundreds of frames, so naive per-frame summing overcounts by orders of
magnitude. Each detection carries a persistent ByteTrack ID and the
inventory aggregates **per track**, so one physical item yields exactly one
row. Per track: confidence-weighted majority vote over object class,
material classified once on the largest (closest) crop, MC-Dropout
uncertainty, OCI score, and routing decision.

```bash
edgewaste-video --source beach_survey.mp4 \
    --det-ckpt runs/detect/taco_multiclass/weights/best.pt \
    --cls-ckpt runs/stage1/best.pt
```

Outputs `annotated.mp4`, `tracks.csv` (one row per physical item) and
`inventory.txt` (counts by object type, material and routing decision, plus
any object/material contradictions).

## What's built (Stage 2/3, hardware simulated)

| Module | File | Simulated? |
|---|---|---|
| Organic Contamination Index (fit / ablate / threshold) | [src/edgewaste/oci/](src/edgewaste/oci/) | Fitted on synthetic calibration data ([oci/synthetic.py](src/edgewaste/oci/synthetic.py)); math + fitting code is real |
| Simulated moisture/gas/metal/load-cell sensors | [src/edgewaste/sensors.py](src/edgewaste/sensors.py) | Yes — no physical sensors exist |
| MC-Dropout Bayesian uncertainty | [src/edgewaste/confidence.py](src/edgewaste/confidence.py) | No — real stochastic-pass estimation on the trained classifier |
| Decision engine + mock conveyor actuator | [src/edgewaste/decision.py](src/edgewaste/decision.py) | Actuation is logged, not driven to GPIO |
| Grad-CAM explainability (ConvNeXt branch) | [src/edgewaste/explain.py](src/edgewaste/explain.py) | No — real Grad-CAM; ViT branch still uncovered |
| Per-detection CSV logging | [src/edgewaste/logging_utils.py](src/edgewaste/logging_utils.py) | No |
| Federated learning (FedAvg simulation) | [src/edgewaste/federated/](src/edgewaste/federated/) | Single-process simulation, not physically distributed |
| ONNX export + parity check | [src/edgewaste/export_onnx.py](src/edgewaste/export_onnx.py) | No — real export; on-Pi benchmark still pending hardware |
| Full pipeline integration | [src/edgewaste/pipeline.py](src/edgewaste/pipeline.py), [run_camera.py](run_camera.py) | Ties all of the above together per-frame |

```bash
# OCI: fit + ablate + threshold on synthetic calibration data, print scores
# under all three sensor-availability cases (both / moisture-only / gas-only)
python scripts/fit_oci_demo.py
python scripts/fit_oci_demo.py --csv path/to/real_calibration_data.csv  # once hardware exists

# Federated learning: single-machine FedAvg simulation across virtual clients
edgewaste-federated --config configs/stage1.yaml --num-clients 4 --rounds 3 --local-epochs 1
edgewaste-federated --config configs/stage1.yaml --smoke   # tiny fast wiring check

# ONNX export with PyTorch-parity verification (classifier + optionally detector)
edgewaste-export --ckpt runs/stage1/best.pt --out runs/stage1/model.onnx \
    --det-ckpt runs/detect/taco_single_class/weights/best.pt --det-out runs/detect/model.onnx

# Full pipeline: detect -> classify -> MC-Dropout confidence -> simulated
# sensors -> OCI -> decision engine -> mock conveyor actuation, all logged.
# 'e' in camera mode saves a Grad-CAM heatmap for the last frame's detections.
python run_camera.py --det-ckpt runs/detect/taco_single_class/weights/best.pt \
    --cls-ckpt runs/stage1/best.pt
edgewaste-pipeline --det-ckpt runs/detect/taco_single_class/weights/best.pt \
    --cls-ckpt runs/stage1/best.pt --log-dir runs/logs/manual_session path/to/images
```

Install the extras: `pip install -e ".[detect,explain,export]"` (or see
`requirements.txt`).

## Setup

```bash
# (recommended) create a venv first
pip install -e .

# GPU: install a CUDA build of torch/torchvision BEFORE the line above, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# The code auto-detects CUDA/MPS/CPU — no code change needed.
```

## Stage 1 workflow

```bash
# 1. See the taxonomy and where each source maps
python -m edgewaste.taxonomy

# 2. Download the three public classification sources (Kaggle token required
#    at ~/.kaggle/kaggle.json; see kaggle_sources.py's CREDS_HELP for setup).
edgewaste-fetch --config configs/stage1.yaml

# 3. Consolidate all available sources into data/processed/<class>/
edgewaste-ingest --config configs/stage1.yaml

# 4. Build the stratified split manifest (data/splits.csv)
edgewaste-split --config configs/stage1.yaml

# 5. Validate the whole pipeline fast (1 epoch, 64-image subset)
edgewaste-train --config configs/stage1.yaml --smoke

# 6. Full training
edgewaste-train --config configs/stage1.yaml

# 7. Evaluate on the held-out test split (writes metrics + confusion matrix)
edgewaste-eval --config configs/stage1.yaml --ckpt runs/stage1/best.pt

# 8. Classifier-only inference demo (no detector — single pre-framed image)
edgewaste-infer --ckpt runs/stage1/best.pt path/to/image_or_dir   # files
edgewaste-infer --ckpt runs/stage1/best.pt --camera               # live webcam
```

## Detection stage workflow (YOLO26)

```bash
# 1. Download TACO (Kaggle YOLO-format mirror) and collapse to a single
#    'waste_item' class, writing the standard Ultralytics dataset layout.
edgewaste-detect-fetch --config configs/detect.yaml

# 2. Fine-tune YOLO26n for item localization
edgewaste-detect-train --config configs/detect.yaml

# 3. Run the full detect-then-classify pipeline
edgewaste-pipeline --det-ckpt runs/detect/taco_single_class/weights/best.pt \
    --cls-ckpt runs/stage1/best.pt path/to/image_or_dir
edgewaste-pipeline --det-ckpt runs/detect/taco_single_class/weights/best.pt \
    --cls-ckpt runs/stage1/best.pt --camera
```

Install the detection extra first: `pip install -e ".[detect]"` (or
`pip install ultralytics` directly — see the version caveat in
`requirements.txt`, since YOLO26 was a January 2026 release).

## The taxonomy (7 classes)

**Decision (supersedes the 2026-07-18 "merge 18-class" decision):** target the
tech-stack/action-plan docs' plain recycling taxonomy — `cardboard, paper,
plastic, glass, metal, organic, other` — sourced entirely from public
datasets (TrashNet, TrashBox, Garbage-Classification-12; see `taxonomy.py`).
The custom 18-class medical/PPE dataset (mask, glove, syringe, IV bag, cotton,
battery, ...) that Stage 1 previously trained toward is **out of scope** for
this taxonomy and is no longer referenced by the pipeline.

**Known gap:** ZeroWaste-f and WaRP-C (the tech-stack doc's other named
detection/localization sources, alongside TACO) have no confirmed Kaggle
mirror as of 2026-07 and require manual download from their project pages —
not wired into `SOURCES`/`DetectConfig` yet. See `ZEROWASTE_F_GAP` in
`taxonomy.py`.

## Notes & constraints

- **Disk:** each checkpoint is ~200 MB (50.7M trainable params). This machine
  was at 99% full during development — keep an eye on free space; checkpoints
  write atomically so a full disk won't corrupt an existing `best.pt`.
- **Model size:** defaults are `convnext_tiny` + `vit_small_patch16_224` to fit a
  4 GB GPU / run on CPU. Bump to `convnext_small` + `vit_base_patch16_224` in
  `configs/stage1.yaml` on stronger hardware for the blueprint's nominal pairing.
- **Feature hand-off:** `HybridConvNeXtViT.feature_vector()` exposes the fused
  vision vector — the intended input to Stage 2's sensor-fusion layer.

## Data & artifacts are git-ignored

`data/`, `runs/`, checkpoints, and datasets are excluded via `.gitignore`. Only
code, configs, and docs are tracked.
