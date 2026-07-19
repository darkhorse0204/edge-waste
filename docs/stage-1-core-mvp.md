# Stage 1 — Core MVP

> Goal: a working camera-only waste classifier. No sensors, no fusion, no FL yet.
> This is the foundation everything else in Stages 2–3 builds on.

> **Implementation status (2026-07-18):** the full Stage 1 pipeline is built and
> validated end-to-end (ingest → split → train → evaluate → infer) on the custom
> dataset. Taxonomy decision resolved to the **merged 18-class** set (see PLAN.md
> "Decisions Made"). Remaining before exit: download the public datasets to fill
> Organic/E-Waste/General-Waste/Shoes (needs a Kaggle token), run a full training
> job (not just the smoke test), and record real metrics. Code lives under
> `src/edgewaste/`; see the repo README for commands.

## Scope (per blueprint)

"Camera + dataset + Transformer/ConvNeXt model + basic waste classification."

Maps to blueprint **Module 1**, **Module 2**, and **Module 3**, plus the general
**AI Pipeline** (section 7) and the **12 waste classes** (section 12).

## What needs to be done

### 1. Dataset Collection (Module 1)
- [~] Source and download public datasets — download tooling built
      (`edgewaste-fetch`, Kaggle API) targeting Garbage-Classification-12 and an
      E-Waste set; **not yet run** (needs a Kaggle token). TACO/TrashNet not wired
      (the 12-class + e-waste sets cover the missing blueprint classes better).
- [x] Custom dataset already exists on disk — inspected 2026-07-16, see
      **"Custom dataset audit"** below for what's actually in it and the
      taxonomy conflict it raises.
- [x] Consolidate into the target classes — done via `edgewaste-ingest`, driven
      by per-source raw→canonical mappings in `src/edgewaste/taxonomy.py`. Custom
      dataset consolidated: 8,700 images across 14 of the 18 canonical classes;
      the 4 public-only classes stay empty until `edgewaste-fetch` is run.
- [x] Target taxonomy decided: **merged 18-class** set (see PLAN.md). Class
      imbalance handled at train time via inverse-frequency class weights +
      a weighted sampler (`WasteDataset.class_weights` / `sampler_weights`).

#### Custom dataset audit

Location: `C:\Users\91738\Desktop\sidequest\Waste Classification Project\Waste Classification Project\Dataset\Custom Dataset Multiclass\custom_dataset_multiclass\custom_dataset_multiclass`
(a `.zip` of the same folder sits one level up; no other public datasets are
present in that `Dataset\` directory yet — TACO/TrashNet/Garbage Classification
still need to be sourced separately).

Flat `ImageFolder`-style layout, no `data.yaml`/annotations — 14 class folders,
~8,686 `.jpg` files total:

| Class | Files |
|---|---|
| battery | 666 |
| biodegradable | 666 |
| cardboard | 666 |
| cotton | 640 |
| glass | 666 |
| glove | 651 |
| I.V | 581 |
| mask | 666 |
| metal | 666 |
| paper | 666 |
| plastic | 666 |
| styrofoam | 666 |
| syringe | 562 |
| textile | 272 |

Two things worth flagging rather than assuming past:

1. **This doesn't look like hand-captured "own" photos.** Every filename
   matches the pattern `<name>_jpg.rf.<32-char-hash>.jpg` (or `_png.rf.…`) —
   the `.rf.` infix is Roboflow's export signature. Several classes also have
   multiple files sharing the same base name before `.rf.` (e.g.
   `cardboard100_jpg.rf.<hash1/2/3>.jpg`), meaning one source image was
   exported multiple times (augmented copies), not that there are that many
   unique photos. Image resolution is also inconsistent *between* classes —
   mostly 640×640, but `battery` samples are 225×225/275×183 and
   `biodegradable`/`styrofoam` are 416×416 — consistent with several
   different Roboflow projects/datasets having been merged by class, not one
   uniform capture session. Treat this as an aggregated/curated dataset, not
   literally "own" images per the blueprint's Module 1 wording — worth
   confirming with whoever assembled it before citing it as original data
   in the report.
2. **Class taxonomy doesn't match the blueprint's list at all.** Blueprint
   Module 12 names: Organic, Plastic, Paper, Glass, Metal, Cardboard,
   Textile, E-Waste, Hazardous, General Waste. This dataset instead skews
   toward medical/PPE waste: `I.V`, `glove`, `mask`, `syringe`, `battery`,
   `cotton`, alongside general recyclables (`cardboard`, `glass`, `metal`,
   `paper`, `plastic`, `styrofoam`, `textile`, `biodegradable`). There's
   partial overlap (cardboard/glass/metal/paper/plastic/textile) but no
   Organic/E-Waste/Hazardous/General-Waste classes, and the medical items
   have no equivalent in the blueprint's list at all.
   **Open decision, not yet resolved:** either (a) treat this as a
   medical/biohazard-waste variant of the project and adjust the 12-class
   target list to match what's actually available, or (b) keep the
   blueprint's original class list and use this dataset only for the
   overlapping classes, sourcing the rest (Organic, E-Waste, Hazardous,
   General Waste) from TACO/TrashNet/Garbage Classification. This changes
   the classifier head's output space, so it should be settled before
   Module 3 training starts.
3. `textile` (272) and `I.V`/`syringe` (581/562) are noticeably smaller than
   the ~650–666 baseline — class imbalance to handle (oversampling/class
   weights) once the final class list is settled.

### 2. Data Preprocessing (Module 2)
- [x] Resize to a consistent input resolution (224×224, configurable).
- [x] Normalization (ImageNet mean/std — matches both pretrained backbones).
- [x] Augmentation: rotation, brightness, random resized crop, blur, random
      erasing (see `data/dataset.py::build_transforms`).
- [x] Stratified train/val/test split via `edgewaste-split` (val + test, per the
      validation stage in the AI Pipeline below). Rare classes kept in every split.

### 3. Vision Model (Module 3)
- [x] Implement ConvNeXt backbone (timm, pretrained, pooled feature vector).
- [x] Implement Vision Transformer backbone (timm, pretrained).
- [x] Add an attention layer combining both into a single feature vector
      (`models/hybrid.py::AttentionFusion` — learned softmax weights over the two
      projected backbone features, then concat → classifier head).
- [x] Architecture confirmed as the hybrid ConvNeXt + ViT + attention design.

### 4. Basic Classification Output
- [x] Classifier head over the fused feature vector (`HybridConvNeXtViT.head`).
- [x] Output space = **18 merged classes** (taxonomy decision, see PLAN.md),
      superseding the blueprint's ambiguous "10 named / 12 classes" list. Head
      size is driven by `taxonomy.NUM_CLASSES`, so it tracks the taxonomy.

### 5. AI Pipeline (section 7, Stage-1-relevant portion)
- [x] Dataset → Training → Validation → Testing implemented
      (`train.py` + `evaluate.py`); checkpoint export = `runs/stage1/best.pt`
      with the class list embedded.
- [~] Deploy to edge device — inference entry point built (`infer.py`, files +
      webcam modes); on-device run pending hardware choice. Full edge loop
      (sensors, fusion, decision, motor) is Stage 2.

## Hardware needed for this stage
- Camera: Raspberry Pi Camera or USB Camera.
- Processing: Jetson Nano or Raspberry Pi 5 (decision needed — see PLAN.md open questions).

## Software needed for this stage
- Python, PyTorch, OpenCV, NumPy, Scikit-learn.
- Ultralytics (if a YOLO-based path is explored for detection, though core
  Stage 1 model per blueprint is ConvNeXt/ViT, not YOLO).

## Exit criteria for Stage 1

Before moving to Stage 2, there should be:
- A trained model that classifies waste images into the target classes with
  a documented accuracy/precision/recall (metrics list is in blueprint section 14).
- A model export usable for on-device inference (format TBD based on hardware choice).
- A basic inference demo running on the chosen edge device using the camera only.

## Explicitly NOT in Stage 1
- Any of the sensors (moisture, gas, metal, weight) — Stage 2.
- OCI, confidence estimation, decision engine, conveyor belt — Stage 2.
- Federated Learning, SHAP/LIME explainability — Stage 3.
