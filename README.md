# edge-waste

Edge-deployed waste classification with sensor-augmented vision and federated
learning. See [PLAN.md](PLAN.md) and [docs/](docs/) for the full staged blueprint.

**Status:** Stage 1 (camera-only classifier) is implemented and wired end-to-end.
Stages 2–3 (sensor fusion, OCI, confidence, conveyor control, federated learning,
explainability) are still design-only in `docs/`.

## What's built (Stage 1)

A **ConvNeXt + Vision Transformer hybrid** image classifier with attention-based
feature fusion, over a **merged 18-class taxonomy** that combines the custom
on-disk dataset with public datasets (per the 2026-07-18 taxonomy decision — see
[taxonomy.py](src/edgewaste/taxonomy.py) and PLAN.md).

```
raw sources ──ingest──► data/processed/<class>/  ──split──► data/splits.csv
                                                                  │
                                          train ◄────────────────┘
                                            │
                                     runs/stage1/best.pt ──► evaluate / infer
```

| Module | File |
|---|---|
| Canonical taxonomy + source mappings | [src/edgewaste/taxonomy.py](src/edgewaste/taxonomy.py) |
| Config (YAML → dataclasses) | [src/edgewaste/config.py](src/edgewaste/config.py), [configs/stage1.yaml](configs/stage1.yaml) |
| Consolidate sources → canonical classes | [src/edgewaste/data/ingest.py](src/edgewaste/data/ingest.py) |
| Download public datasets (Kaggle) | [src/edgewaste/data/kaggle_sources.py](src/edgewaste/data/kaggle_sources.py) |
| Stratified train/val/test split | [src/edgewaste/data/splits.py](src/edgewaste/data/splits.py) |
| Dataset + augmentations | [src/edgewaste/data/dataset.py](src/edgewaste/data/dataset.py) |
| ConvNeXt+ViT hybrid model | [src/edgewaste/models/hybrid.py](src/edgewaste/models/hybrid.py) |
| Training loop | [src/edgewaste/train.py](src/edgewaste/train.py) |
| Evaluation + confusion matrix | [src/edgewaste/evaluate.py](src/edgewaste/evaluate.py) |
| Inference demo (files + webcam) | [src/edgewaste/infer.py](src/edgewaste/infer.py) |

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
# 1. See the merged taxonomy and where each source maps
python -m edgewaste.taxonomy

# 2. (optional) download public datasets to fill Organic / E-Waste / General
#    Waste / Shoes. Needs a Kaggle API token (~/.kaggle/kaggle.json). Skips
#    cleanly if credentials are absent — the pipeline still runs on the custom
#    dataset's 14 classes.
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

# 8. Inference demo
edgewaste-infer --ckpt runs/stage1/best.pt path/to/image_or_dir   # files
edgewaste-infer --ckpt runs/stage1/best.pt --camera               # live webcam
```

## The merged taxonomy (18 classes)

Chosen resolution to the custom-vs-blueprint class mismatch: **merge both.**
Custom dataset supplies the 14 medical/recycling classes on disk; public
datasets fill the blueprint-only classes.

- **recyclable:** cardboard, paper, plastic, glass, metal, textile, styrofoam, shoes
- **organic:** organic (custom `biodegradable` + public `biological`)
- **medical:** mask, glove, syringe, iv_bag, cotton *(custom-only)*
- **special:** battery, e_waste, hazardous, general_waste

**Known data gap:** `hazardous` has no clean public source mapped yet, so it
stays empty (the trainer warns and weights it out) until a source is added or
it's folded into `battery`/`e_waste`. See `HAZARDOUS_GAP` in `taxonomy.py`.

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
