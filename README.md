# edge-waste

Edge-deployed waste classification with sensor-augmented vision and federated
learning. See [PLAN.md](PLAN.md) and [docs/](docs/) for the full staged blueprint,
and the three planning docs at the repo root (tech-stack recommendation, project
action plan, OCI calibration protocol) for the converged design this now follows.

**Status:** Stage 1 (detect-then-classify vision pipeline) is implemented and
wired end-to-end. Stage 2 (sensors, OCI, confidence, conveyor control) and
Stage 3 (federated learning, explainability) are still design-only in `docs/`.

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
