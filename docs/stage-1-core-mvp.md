# Stage 1 — Core MVP

> Goal: a working camera-only waste classifier. No sensors, no fusion, no FL yet.
> This is the foundation everything else in Stages 2–3 builds on.

> **Implementation status (2026-07-20):** the full Stage 1 pipeline is built
> (ingest → split → train → evaluate → infer), plus a new detect-then-classify
> path (`edgewaste.detect` + `edgewaste.pipeline`). Taxonomy decision now
> resolved to the **7-class recycling-only** set, superseding the 2026-07-18
> merged-18-class decision below (see PLAN.md "Decisions Made") — the custom
> dataset audit that follows is kept for history but no longer reflects what
> the pipeline trains on. Remaining before exit: download the three public
> Kaggle sources (needs a Kaggle token), run a full training job on the new
> taxonomy (not just the smoke test), fine-tune the YOLO26 detector on TACO,
> and record real metrics for both. Code lives under `src/edgewaste/`; see the
> repo README for commands.

## Scope (per blueprint)

"Camera + dataset + Transformer/ConvNeXt model + basic waste classification."

Maps to blueprint **Module 1**, **Module 2**, and **Module 3**, plus the general
**AI Pipeline** (section 7) and the **12 waste classes** (section 12).

## What needs to be done

### 1. Dataset Collection (Module 1)
- [~] Source and download public datasets — download tooling built
      (`edgewaste-fetch`, Kaggle API) targeting TrashNet, TrashBox, and
      Garbage-Classification-12 (verified Kaggle mirrors, see
      `taxonomy.py`); **not yet run** (needs a Kaggle token).
- [x] Consolidate into the target classes — `edgewaste-ingest`, driven by
      per-source raw→canonical mappings in `src/edgewaste/taxonomy.py`, walks
      the whole downloaded tree matching on leaf folder name (robust to
      whatever nesting each dataset zip unpacks to).
- [x] Target taxonomy decided (2026-07-20): **7-class recycling-only** set —
      cardboard, paper, plastic, glass, metal, organic, other (see PLAN.md
      "Decisions Made"). Supersedes the merged-18-class decision below. Class
      imbalance handled at train time via inverse-frequency class weights +
      a weighted sampler (`WasteDataset.class_weights` / `sampler_weights`).
- [x] Detection dataset (TACO, single-class) wired separately via
      `edgewaste.detect.data` — see Module 3 below and `docs/` root README.

#### Custom dataset audit — historical, no longer current (kept for record)

> The custom dataset described below was the basis for the superseded
> 2026-07-18 18-class taxonomy decision. It is **not used** by the current
> 7-class pipeline; this section is kept only so the reasoning behind
> abandoning it stays on record.

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

### 3. Vision Model (Module 3) + detection stage (added 2026-07-20)
- [x] Implement ConvNeXt backbone (timm, pretrained, pooled feature vector).
- [x] Implement Vision Transformer backbone (timm, pretrained).
- [x] Add an attention layer combining both into a single feature vector
      (`models/hybrid.py::AttentionFusion` — learned softmax weights over the two
      projected backbone features, then concat → classifier head).
- [x] Architecture confirmed as the hybrid ConvNeXt + ViT + attention design.
- [x] **New:** detect-then-classify pipeline per the tech-stack doc §2.3 —
      YOLO26 (single-class `waste_item` localizer, `edgewaste.detect`) finds
      and crops items before the classifier runs on each crop
      (`edgewaste.pipeline`). Not yet trained/run — code only.

### 4. Basic Classification Output
- [x] Classifier head over the fused feature vector (`HybridConvNeXtViT.head`).
- [x] Output space = **7 classes** (taxonomy decision, see PLAN.md), narrowed
      from the earlier 18-class merge. Head size is driven by
      `taxonomy.NUM_CLASSES`, so it tracks the taxonomy automatically.

### 5. AI Pipeline (section 7, Stage-1-relevant portion)
- [x] Dataset → Training → Validation → Testing implemented
      (`train.py` + `evaluate.py`); checkpoint export = `runs/stage1/best.pt`
      with the class list embedded.
- [x] Detector training implemented (`edgewaste.detect.train`, ultralytics),
      checkpoint export = `runs/detect/taco_single_class/weights/best.pt`.
- [~] Deploy to edge device — inference entry points built (`infer.py`
      classifier-only; `pipeline.py` detect+classify; both support files +
      webcam); on-device run pending hardware choice. Full edge loop
      (sensors, fusion, decision, motor) is Stage 2.

## Hardware needed for this stage
- Camera: Raspberry Pi Camera or USB Camera.
- Processing: laptop/Colab GPU is enough for training both models; defer a
  Jetson/Pi 5 purchase until CPU-only throughput is confirmed insufficient
  (per `Project-Action-Plan.md`).

## Software needed for this stage
- Python, PyTorch, timm, OpenCV, NumPy, Scikit-learn (classifier).
- Ultralytics (`pip install -e ".[detect]"`) for the YOLO26 detector — now
  actually used, not just a maybe.

## Exit criteria for Stage 1

Before moving to Stage 2, there should be:
- A trained detector that localizes items with a documented mAP on a held-out
  TACO split, and a trained classifier that types crops into the 7 target
  classes with documented accuracy/precision/recall (metrics list is in
  blueprint section 14).
- A model export usable for on-device inference (format TBD based on hardware choice).
- A basic inference demo running on the chosen edge device using the camera only.

## Explicitly NOT in Stage 1
- Any of the sensors (moisture, gas, metal, weight) — Stage 2.
- OCI, confidence estimation, decision engine, conveyor belt — Stage 2.
- Federated Learning, SHAP/LIME explainability — Stage 3.
