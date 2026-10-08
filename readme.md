<!-- readme.md - the single universal reference for the project: data, model, training, every result, analysis, how to run, changelog -->

# edge-waste — CAPS-FCL waste classification

An edge-deployable waste-sorting system: a camera finds each item, a
ConvNeXt + Vision-Transformer hybrid identifies it (33 item classes in 9
material families, including hazardous batteries, e-waste and medical waste),
moisture/gas sensors score organic contamination, and a decision engine routes
the item to a bin — with calibrated uncertainty, a statistically bounded
hazard-leakage rate, explanations, and federated learning across units.

> **This readme is the one document to read.** It is updated with every
> change to the project (see the [changelog](#changelog)). Every number in it
> comes from a logged run in this repository; commands to regenerate each one
> are in [Reproducibility](#reproducibility).

**Status (TRL 3, experimental proof of concept).** All software is built and
evaluated on real held-out data. Physical sensors, the diverter actuator and
Raspberry Pi deployment do not exist yet — sensors are simulated and the
contamination model is fitted on synthetic calibration data (clearly marked
wherever it appears).

## Headline results (held-out test set, 4,232 images never used in training or model selection)

| Metric | Result | 95% bootstrap CI |
|---|---|---|
| Item accuracy (33 classes) | **89.30%** | 88.42 – 90.19% |
| Family / routing accuracy (9 bins) | **94.14%** | 93.43 – 94.80% |
| Hazardous-item recall | **93.50%** (690/738) | 91.63 – 95.13% |
| Macro F1 | 0.863 | 0.850 – 0.874 |
| Top-3 accuracy | 96.74% | |
| Macro ROC-AUC (one-vs-rest) | 0.981 | |
| Detector mAP50 (1-class / 18-class TACO) | 0.700 / 0.383 | |

The table above is the first ConvNeXt + ViT hybrid (`runs/stage1`), the model the
rest of the analysis is based on. The analysis found that an RBF SVM on *frozen*
ImageNet ConvNeXt features did better (93.3% item, 97.1% family, 98.1% hazard
recall, McNemar p ≈ 8e-21), which exposed a fine-tuning problem. The retrain
with the fixed recipe (`runs/stage1_v2`) is the **current best model**:

| v2 (Colab-reported, 4,232 test images) | Result |
|---|---|
| Item accuracy | **93.83%** |
| Family / routing accuracy | **97.28%** |
| Hazardous recall | **98.37%** |
| Best validation accuracy | 93.86% (epoch 8 of 12) |

v2's numbers are the ones Colab measured on its own held-out split; they have
**not yet been independently reproduced locally** (see
[Improving the classifier](#improving-the-classifier) for why, and how). Its
bootstrap intervals, calibration and baselines are still those of the first model.

## Contents

1. [Folder structure](#folder-structure)
2. [Quick start](#quick-start)
3. [Data](#data)
4. [Model architecture](#model-architecture)
5. [Training setup](#training-setup)
6. [Results](#results)
7. [Overfitting and underfitting](#overfitting-and-underfitting)
8. [Calibration](#calibration)
9. [Class imbalance](#class-imbalance)
10. [Classical ML baselines: PCA, LDA, SVM, kNN, logistic regression, random forest, naive Bayes](#classical-ml-baselines)
11. [What the model learned: embeddings and backbone attention](#what-the-model-learned)
12. [Improving the classifier (findings and the next training run)](#improving-the-classifier)
13. [Uncertainty and risk-bounded hazard routing](#uncertainty-and-risk-bounded-hazard-routing), including the [Simulations](#simulations) that test it
14. [Object detection](#object-detection)
15. [Contamination index (sensor fusion)](#contamination-index-oci)
16. [Decision engine](#decision-engine)
17. [Explainability, segmentation, augmentation, federated learning, video, deployment](#other-components)
18. [Model cost](#model-cost)
19. [Reproducibility](#reproducibility)
20. [Limitations](#limitations)
21. [Review FAQ](#review-faq)
22. [Changelog](#changelog)

## Folder structure

```
edge-waste/
├── readme.md                         this document
├── project_status.md                 dated session log and open risks
├── pyproject.toml, requirements.txt  package + dependencies (edgewaste-* commands)
├── configs/                          classifier_convnext_vit.yaml (main), classifier_convnext_swin.yaml,
│                                     classifier_professor_dataset.yaml, detector_yolo_taco.yaml
├── notebooks/                        colab_classifier_training.ipynb (GPU training on Colab)
├── scripts/                          launchers, demos, analysis drivers, detector training helpers
│   ├── prototype_sorter.py, calibrate_prototype_oci.py   run and calibrate the hardware prototype (mock mode needs no hardware)
│   └── report/                       builds the project report (word) and the 25-slide review deck
├── reports/ml_analysis/              generated analysis: summary.json, CSV tables, figures/
├── reports/simulations/              simulations of the claimed mechanisms: summary.json, figures/
├── src/edgewaste/
│   ├── config.py, taxonomy.py, common_utils.py
│   ├── data/                         download, ingest, split, load images
│   ├── classification/               hybrid model, train, evaluate, inference, onnx export
│   ├── detection/                    yolo detector, coco object recogniser, sam segmentation
│   ├── decision_engine/              identity prior, mc-dropout, conformal routing, routing rules
│   ├── contamination/                oci normalisation, models, simulated sensors, shap
│   ├── explainability/               grad-cam, lime
│   ├── augmentation/                 dcgan synthetic images
│   ├── federated/                    fedavg simulation
│   ├── analysis/                     metrics, fit diagnostics, calibration, imbalance, baselines, cost
│   └── applications/                 live camera pipeline, video litter survey
├── hardware/arduino/sorter_node/     arduino firmware of the 2-day prototype (sensors, tilt servo, leds, buzzer)
├── hardware/esp32/sorter_node_esp32/ esp32 firmware (dht22 humidity as moisture channel, mq-135, servo, boot button)
├── docs/esp32_prototype_walkthrough.md   step-by-step build with esp32 + dht22 + mq-135 + servo
├── docs/planning/                    original plans and stage checklists (historical)
├── docs/hardware_prototype_plan.md   two-day minimum-cost prototype: parts, wiring, build, calibration, schedule, tests
├── docs/faculty_reviews/             review submissions (not in git)
├── docs/report/                      BITE497J Project I report (docx, pdf), review slides (pptx), figures, videos (outputs not in git; mermaid sources are)
├── weights/                          base yolo26n / mobile sam weights (not in git)
└── data/, runs/                      datasets, checkpoints, caches (not in git)
```

Every source, config and doc file starts with a one-line lowercase comment
stating what it is.

## Quick start

```bash
pip install -e ".[data,detect,explain,export]"   # CUDA torch first if you have a GPU

python -m edgewaste.taxonomy                      # the 33 classes and where each comes from
edgewaste-fetch  --config configs/classifier_convnext_vit.yaml   # Kaggle token needed
edgewaste-ingest --config configs/classifier_convnext_vit.yaml
edgewaste-split  --config configs/classifier_convnext_vit.yaml
edgewaste-train  --config configs/classifier_convnext_vit.yaml   # or notebooks/ on Colab
edgewaste-eval   --config configs/classifier_convnext_vit.yaml --ckpt runs/stage1/best.pt \
                 --manifest data/splits_colab_reconstructed.csv

python scripts/run_live_camera.py                 # live detect -> classify -> route demo
edgewaste-video --source clip.mp4 --det-ckpt <detector best.pt> --cls-ckpt runs/stage1/best.pt
```

## Data

### Sources

| Source (Kaggle) | Used for | Images |
|---|---|---|
| Recyclable and Household Waste Classification (`alistairking/...`) | 30 item classes, studio + real-world photos | 15,000 |
| Garbage Classification 12 (`mostafaabla/garbage-classification`) | battery, clothing, shoes, extra food waste | |
| TrashBox (`minhle13/trashbox`) | e-waste, medical — the hazardous classes no other source has | |

Coarse folders that cannot be mapped to one fine-grained class (e.g. a generic
"plastic" folder) are deliberately dropped rather than guessed. Unreadable
images are rejected at ingest (one corrupt TrashBox e-waste JPEG).

### Taxonomy: 33 item classes in 9 material families

The classifier predicts the **item**; the **family** (= physical bin) is
derived from it. Numbers are training images / test F1.

| Family | Items |
|---|---|
| plastic (4,500) | water bottles 350/0.83, soda bottles 350/0.85, detergent bottles 350/0.95, food containers 350/0.90, shopping bags 350/0.90, trash bags 350/0.90, cup lids 350/0.89, straws 350/0.89, cutlery 350/0.94 |
| paper (2,000) | newspaper 350/0.79, magazines 350/0.91, office paper 350/0.71, paper cups 350/0.84 |
| cardboard (1,000) | boxes 350/0.61, packaging 350/0.56 |
| glass (1,500) | beverage bottles 350/0.90, food jars 350/0.94, cosmetic containers 350/0.93 |
| metal (2,000) | aluminium soda cans 350/0.89, aluminium food cans 350/0.62, steel food cans 350/0.62, aerosol cans 350/0.95 |
| organic (2,985) | food waste 1,039/0.95, eggshells 350/0.91, coffee grounds 350/0.97, tea bags 350/0.87 |
| styrofoam (1,000) | cups 350/0.93, food containers 350/0.89 |
| textile (8,302) | clothing 4,077/0.96, shoes 1,733/0.93 |
| **hazardous** (4,916) | battery 661/0.95, e-waste 1,684/0.91, medical 1,095/0.87 |

Why a hierarchy: a sorting facility acts on the family, so family accuracy is
the operational metric, and item-level errors inside one family (aluminium vs
steel food cans) are recognition errors with no routing consequence. Hazardous
items are separated at the taxonomy level so their recall is never averaged
away.

### Splits and leakage prevention

- **Stratified 70 / 15 / 15** per class, seed 42: **19,739 train / 4,232 val /
  4,232 test** (28,203 manifest rows including the one corrupt image, which the
  loader skips).
- **Validation** is used for early stopping and every hyperparameter or
  threshold choice; **test** is scored once per model.
- **Duplicate-safe file names:** each processed image is named by a hash of
  its source path, so the same file never appears twice.
- **A leakage bug was found and fixed.** The hash was originally taken over
  the OS-specific path string, so Windows and Colab produced different names,
  different sort orders and therefore different splits: re-scoring the
  Colab-trained checkpoint on a locally rebuilt split gave an inflated 94.52%
  because ~70% of its "test" images had been training images. It was caught
  because the number went *up*. Hashing now uses the POSIX path relative to
  the source root.
- **The exact Colab split was recovered** (`scripts/reconstruct_colab_split.py`)
  by recomputing the old Colab-side names and re-adding the one corrupt image
  that Colab's split had counted. Verified: the checkpoint scores exactly
  3,779/4,232 = 89.30%, 94.14% family and 690/738 hazard recall on it — all
  three Colab-reported numbers. Every analysis below uses this split.

### Preprocessing and augmentation

- Resize to 224x224, ImageNet mean/std normalisation (matches backbone pretraining).
- Training-only augmentation: resize to 256 then RandomResizedCrop 224 (scale
  0.7–1.0), horizontal flip, rotation ±20°, colour jitter (brightness /
  contrast / saturation 0.2), Gaussian blur (p = 0.2), random erasing (p = 0.1)
  — simulating camera pose, lighting and partial occlusion on a conveyor.
- Synthetic images for rare classes are available via a per-class DCGAN
  ([augmentation](#other-components)); not used in the reported model.

## Model architecture

```
image 224x224x3
   ├── ConvNeXt-Tiny (CNN)  -> global avg pool 768-d -> Linear -> LayerNorm -> GELU -> 512-d  f_cnx
   └── ViT-Small/16 (transformer) -> avg pool 384-d  -> Linear -> LayerNorm -> GELU -> 512-d  f_vit
attention fusion:  score_s = W2·GELU(W1·f_s + b1) + b2      alpha = softmax(score_cnx, score_vit)
                   fused   = [alpha_cnx·f_cnx , alpha_vit·f_vit]                            1024-d
head:              Dropout 0.2 -> Linear 1024->512 -> GELU -> Dropout 0.2 -> Linear 512->33
output:            softmax over 33 items; family probability = sum over the family's items
```

- **Why two backbones.** A CNN's local, translation-equivariant bias suits
  surface texture (foil crinkle, cardboard fibre); a ViT attends globally from
  the first layer and suits shape and context. The learned attention decides
  per image how much of each to use — and [measurably](#what-the-model-learned)
  leans on ConvNeXt.
- **Transfer learning.** ConvNeXt-Tiny pretrained on ImageNet-12k then 1k
  (`convnext_tiny.in12k_ft_in1k`); ViT-S/16 AugReg ImageNet-21k then 1k
  (`vit_small_patch16_224.augreg_in21k_ft_in1k`). All layers are fine-tuned.
  Training from scratch was tried once and collapsed to 49% accuracy.
- **Activation functions.** GELU in every hidden layer (both backbones, both
  projections, the fusion scorer, the head); softmax for the fusion weights and
  the output. GELU is what both pretrained backbones use, is smooth, and has no
  dead-neuron problem.
- **Normalisation.** LayerNorm, not BatchNorm — independent of batch
  statistics, so it behaves the same at training batch 32 and edge batch 1.
- **Regularisation in the model.** Two dropout layers (p = 0.2) in the head;
  no stochastic depth.
- **Swin variant.** `configs/classifier_convnext_swin.yaml` swaps the ViT for
  Swin-Tiny (backbone-agnostic code); see [Improving the classifier](#improving-the-classifier).

## Training setup

This is exactly how the reported 89.30% checkpoint (`runs/stage1/best.pt`) was
trained. The main config now holds an improved recipe for the next run — see
[Improving the classifier](#improving-the-classifier).

| Setting | Value | Reason |
|---|---|---|
| Optimiser | **AdamW**, β2 = 0.999, ε = 1e-8, β1 cycled 0.95 → 0.85 → 0.95 by the schedule | per-parameter adaptive steps for two pretrained backbones plus a new head; decoupled weight decay regularises correctly under Adam |
| Learning rate | peak **3e-4**, same for all layers | |
| Schedule | **OneCycle**: cosine warm-up from 1.2e-5 over the first epoch, cosine decay to ~1e-9, stepped every batch | warm-up protects pretrained weights; annealing settles into a flat minimum |
| Weight decay | 0.05 | |
| Loss | cross-entropy, **label smoothing 0.1** | stops over-confident logits between near-duplicate classes |
| Imbalance handling | balanced sampler **and** inverse-frequency loss weights | found to over-correct — see [Class imbalance](#class-imbalance) |
| Batch size / epochs | 32 / 15, early-stopping patience 5 on val accuracy | best epoch was 15 (the last) |
| Precision / clipping | fp16 mixed precision with GradScaler; gradient L2 norm clipped at 1.0 | |
| Seed | 42 | |
| Hardware | Google Colab T4 GPU, ~4 min per epoch | |

## Results

Full tables: `reports/ml_analysis/summary.json`,
`reports/ml_analysis/per_class_metrics_test.csv`.

| Metric (test, n = 4,232) | Value |
|---|---|
| Accuracy / balanced accuracy | 89.30% / 86.78% |
| Macro precision / recall / F1 | 0.860 / 0.868 / 0.863 |
| Weighted F1 | 0.894 |
| Cohen's kappa / Matthews correlation | 0.885 / 0.885 |
| Top-3 / top-5 accuracy | 96.74% / 97.64% |
| Macro ROC-AUC / macro PR-AUC | 0.981 / 0.883 |
| Log loss | 0.651 |
| Family (routing) accuracy | 94.14% |
| Hazardous recall / precision | 93.50% / 95.30% |

The 4.8-point gap between item and family accuracy is the hierarchy working:
**the 7 most frequent confusions are all within one family** — cardboard
packaging ↔ boxes (33 + 21), steel ↔ aluminium food cans (29 + 18), clothing →
shoes (15), e-waste ↔ medical (12 + 10, both hazardous). The weakest classes
are exactly these look-alike pairs (F1 0.56–0.62), not the rare ones.

![per-class precision and recall](reports/ml_analysis/figures/per_class_precision_recall.png)
![confusion matrix, items](reports/ml_analysis/figures/confusion_matrix_items.png)
![confusion matrix, families](reports/ml_analysis/figures/confusion_matrix_families.png)

## Overfitting and underfitting

![fit diagnostics](reports/ml_analysis/figures/fit_diagnostics.png)

| Split (eval mode, no augmentation) | Accuracy | Log loss |
|---|---|---|
| Train (19,739) | 96.89% | 0.379 |
| Validation (4,232) | 89.51% | 0.653 |
| Test (4,232) | 89.30% | 0.651 |

**Verdict: a moderate generalisation gap, not harmful overfitting, and not
underfitting.**

- *Not underfitting:* 96.9% training accuracy — the model has the capacity to fit the data.
- *Some variance:* a 7.6-point train–test gap, expected for 50.7M parameters
  and 19.7k images.
- *Not harmful overfitting:* the best validation accuracy came at the **last**
  epoch (15 of 15), so validation accuracy never turned down — the run was
  stopped by the epoch budget, not by early stopping.
- *No overfitting to the validation set:* val and test differ by only 0.21
  points, so tuning choices made on val transfer to unseen data.
- *Where the gap lives:* the largest per-class gaps are the look-alike pairs
  (cardboard packaging: 82.9% train → 50.7% test; steel food cans 88.0% →
  60.0%). Even their *training* accuracy is low, which points to genuinely
  ambiguous labels rather than memorisation.
- *Learning curve* (logistic regression on frozen features): validation
  accuracy rises from 84.5% at 394 images to 92.3% at 13.8k and is flat from
  13.8k to 19.7k — more data of the same kind now gives diminishing returns.
- *Validation curve:* the same model underfits at C = 1e-4 (88.3% train /
  87.6% val), is best at C = 0.01 (96.0% / 92.4%), and overfits beyond
  (98.9% / 90.9% at C = 100) — the textbook bias–variance picture.

Regularisation used against overfitting: ImageNet pretraining, data
augmentation, dropout, weight decay, label smoothing, early stopping on
validation, and a held-out test set scored once.

The per-epoch curves of the main Colab run live in `runs/stage1/history.json`
on Google Drive; copy it into `runs/stage1/` and re-run
`scripts/run_ml_analysis.py` to plot them (the figure currently shows the
2-epoch Swin run's history).

## Calibration

![reliability diagram](reports/ml_analysis/figures/reliability_diagram.png)

| Test | ECE | Max CE | NLL | Brier | Mean confidence | Accuracy |
|---|---|---|---|---|---|---|
| As trained | 16.56% | 45.1% | 0.651 | 0.232 | 73.0% | 89.30% |
| Temperature-scaled (T = 0.71, fitted on val) | **7.30%** | 32.2% | **0.534** | **0.183** | 89.7% | 89.30% |

The model is **under-confident**: it is right 89% of the time but on average
only 73% sure. That is the known side-effect of label smoothing plus
class-balanced training, both of which pull probabilities toward uniform.
Temperature scaling (dividing logits by T = 0.71) halves the calibration error
without changing a single prediction. Under-confidence matters downstream
because the uncertainty gate sends low-confidence items to human review.

## Class imbalance

![class imbalance](reports/ml_analysis/figures/class_imbalance.png)

- **The imbalance:** 27 classes have 350 training images each; six merged
  classes are larger (clothing 4,077, shoes 1,733, e-waste 1,684, medical
  1,095, food waste 1,039, battery 661). Largest-to-smallest ratio **11.6x**.
- **How it was handled:** a `WeightedRandomSampler` (balanced batches) **and**
  inverse-frequency class weights in the loss, both switched on by one flag.
- **Finding — this double-corrects.** Either mechanism alone makes training
  behave as if all classes were equally common; together, rare classes are
  weighted about 1/n² instead of 1/n. By Bayes' rule this shifts the model's
  posteriors toward rare classes (e.g. office paper was predicted 1.32x as
  often as it occurs).
- **Measured effect, without retraining:** post-hoc logit adjustment (Menon et
  al., ICLR 2021) adds τ·log(n_class) to the logits. τ = 1.75, chosen on
  validation, gives on test: accuracy 89.30% → 89.60%, macro F1 0.863 →
  0.870, **hazard recall 93.50% → 94.72%**, balanced accuracy 0.868 → 0.862.
  τ = 1 (≈ what a single correction would give) reaches 89.86% accuracy and
  94.85% hazard recall. The effect is real but small (under 1 point of
  accuracy), because the imbalance mainly involves six large classes.
- **Fix:** `train.imbalance_strategy` (`sampler` | `loss_weights` | `both` |
  `none`) now applies one correction; the default is `sampler`.

## Classical ML baselines

Question these answer: *does the deep model earn its complexity, and what
do the classical methods reveal?* Protocol (`src/edgewaste/analysis/classical_baselines.py`):
standardise → PCA → classifier; hyperparameters chosen on validation, test
scored once — the same protocol as the deep model.

**Three feature sets**

| Feature set | Raw dim | PCA: components for 90 / 95 / 99% variance | Kept |
|---|---|---|---|
| Hand-crafted: HSV colour histogram (96) + HOG on 64x64 greyscale (1,764) | 1,860 | 439 / 715 / 1,325 | 256 (82.8%) |
| Frozen ImageNet ConvNeXt-Tiny features (no fine-tuning) | 768 | 382 / 513 / 690 | 256 (82.4%) |
| Our fine-tuned hybrid embedding (input to its head) | 1,024 | 23 / **29** / 45 | 29 (95.6%) |

![PCA explained variance](reports/ml_analysis/figures/pca_explained_variance.png)

The fine-tuned embedding packs 95% of its variance into 29 dimensions — close
to the 32 (= classes − 1) that *neural collapse* predicts for a trained
classifier's last layer: fine-tuning compresses the features onto the class
structure.

**Test accuracy (n = 4,232)**

| Classifier | Hand-crafted | Frozen ImageNet | Fine-tuned embedding |
|---|---|---|---|
| Gaussian naive Bayes | 40.5% | 84.5% | 88.5% |
| LDA (Ledoit-Wolf shrinkage) | 47.7% | 89.5% | 89.5% |
| Logistic regression (C by val) | 52.6% | 92.5% | 89.2% |
| Linear SVM (C by val) | 50.8% | 91.9% | **89.8%** |
| RBF-kernel SVM (C by val) | **71.1%** | **93.3%** | 89.2% |
| k-nearest neighbours (k by val) | 61.4% | 91.6% | 89.7% |
| Random forest (300 trees) | 57.1% | 91.5% | 88.9% |
| *Fine-tuned hybrid, own softmax head* | | | *89.3%* |

**Best of each column vs the deep model** (McNemar's paired test on the same test images)

| Model | Accuracy | Macro F1 | Family acc. | Hazard recall | vs hybrid |
|---|---|---|---|---|---|
| Hand-crafted → RBF SVM | 71.1% | 0.676 | 79.4% | 77.4% | hybrid better, p ≈ 5e-135 |
| **Frozen ImageNet → RBF SVM** | **93.3%** | **0.899** | **97.1%** | **98.1%** | **SVM better, p ≈ 8e-21** (fixes 248 hybrid errors, adds 78) |
| Fine-tuned embedding → linear SVM | 89.8% | 0.868 | 94.5% | 94.2% | SVM better, p = 0.0015 |
| Fine-tuned hybrid (reported model) | 89.3% | 0.863 | 94.1% | 93.5% | |

What this shows:

1. **Why deep learning:** the best classical computer-vision pipeline
   (HOG + colour → PCA → RBF SVM) stops at 71%; deep features reach 89–93%.
2. **Bias vs variance in one table:** naive Bayes and LDA have the smallest
   train–test gaps but the lowest accuracy (high bias); random forest and kNN
   reach 99.4% train accuracy but lose 8–42 points on test (high variance);
   the RBF SVM with C tuned on validation sits in between and wins.
3. **The key finding:** an RBF SVM on *un-fine-tuned* ImageNet ConvNeXt
   features beats our fine-tuned hybrid on every metric, significantly. The
   fine-tuning degraded the pretrained features — see
   [Improving the classifier](#improving-the-classifier).
4. **The head is not the bottleneck:** every classifier on the fine-tuned
   embedding lands within 88.5–89.8%, next to the network's own head (89.3%).

LDA is also used as a supervised 2-D projection in
[What the model learned](#what-the-model-learned).

## What the model learned

![embedding projections](reports/ml_analysis/figures/embedding_projections.png)

PCA, LDA and t-SNE projections of the test set coloured by material family.
t-SNE of *frozen* ImageNet features already shows loose family clusters;
after fine-tuning the clusters are compact and mostly separated, with overlap
where the confusion matrix says it should be (cardboard, metal cans, e-waste /
medical).

![fusion attention](reports/ml_analysis/figures/fusion_attention.png)

**Backbone attention:** the fusion layer gives ConvNeXt 82–93% of the weight
for every family (textile 93%, cardboard 92%, paper 82%) and ViT 7–18%. The
model relies mainly on the CNN — texture carries most of the signal for
material recognition. (Attention weight is an indicator, not an exact
attribution: the two projected features can differ in magnitude.)

## Improving the classifier

The analysis points to one cause and one fix.

**Evidence**

- A classifier on frozen ImageNet ConvNeXt features (93.3%) beats the
  fine-tuned hybrid (89.3%) — and the hybrid's ConvNeXt branch started from
  those same weights and receives ~85% of the fusion attention.
- Classical classifiers on the fine-tuned embedding do no better than the
  network's head, so the features, not the head, limit accuracy.
- Validation accuracy was still rising at the last of 15 epochs.
- Class imbalance was corrected twice (small effect, fixed).

**Diagnosis — feature distortion.** All 50M parameters were fine-tuned at the
same peak learning rate (3e-4) from the first step, while the classifier head
was still random. Large, noisy gradients from the random head flow into the
pretrained backbones and overwrite general-purpose features before the head
has learned anything (Kumar et al., *Fine-Tuning can Distort Pretrained
Features*, ICLR 2022). 3e-4 is a head learning rate; pretrained backbones are
normally fine-tuned 10x lower.

**Fix — now the main config** (`configs/classifier_convnext_vit.yaml`):

| Change | Old (reported checkpoint) | New |
|---|---|---|
| Linear-probe-then-fine-tune | no | backbones frozen for the first 2 epochs (`freeze_backbone_epochs`) |
| Discriminative learning rate | 3e-4 everywhere | backbones 3e-5, new layers 3e-4 (`backbone_lr_mult: 0.1`) |
| Imbalance correction | sampler + loss weights | sampler only (`imbalance_strategy`) |
| Epochs | 15 | 25, early stopping patience 5 |
| Output | `runs/stage1` | `runs/stage1_v2` (cannot overwrite the reported checkpoint) |

**Result of the retrain (v2).** 12 epochs on a Colab T4 (~4 min each, 48 minutes
in total), 617 steps per epoch. Per-epoch history is in `runs/stage1_v2/history.json`.

| | v1 (reported) | v2 (this recipe) |
|---|---|---|
| Item accuracy (test) | 89.30% | **93.83%** |
| Family accuracy | 94.14% | **97.28%** |
| Hazardous recall | 93.50% (690/738) | **98.37%** (~726/738) |
| Best val accuracy / epoch | 89.51% / 15 | 93.86% / 8 |
| Training accuracy (augmented) at the end | — | 97.4% |

- Validation accuracy was 90.0% after the **first** epoch (head only, backbones
  frozen) and 93.7% by epoch 5, against 89.5% after 15 epochs for v1 — the
  distortion diagnosis is supported: protecting the pretrained features is worth
  about 4.5 points.
- Validation loss was lowest at epoch 5 (0.830) and rose slowly afterwards while
  training loss kept falling, a mild overfitting trend after epoch 8, as the
  best epoch indicates.
- **v2 is on par with, not clearly better than, the frozen-feature SVM** (93.3%):
  the two were scored on different test splits (the split-hashing fix changed
  which images fall in test), so the 0.5-point difference is inside the
  ±0.9-point sampling noise and is not a paired comparison.
- **Why it is not yet independently verified.** Re-scoring the v2 checkpoint on
  the split rebuilt locally gave 96.8% — higher than Colab's own 93.8% — which is
  the signature of leakage: the local manifest and Colab's contain different
  images, so part of the local "test" set was in Colab's training set. The local
  figure is discarded (the files are kept as `runs/stage1_v2/LEAKY_local_split_*`).
  The check that exposed it: the checkpoint records its validation accuracy
  (93.86%), and the local validation split gives 96.83%. `edgewaste-eval` now
  runs this check automatically and refuses a manifest that does not reproduce
  the training-time validation accuracy. To finish verification, copy
  `data/splits.csv` and `data/processed/provenance.csv` from Google Drive
  (`MyDrive/edge-waste/data/`) and map them onto the local images, as was done
  for v1.

**Available today without retraining:** the frozen ConvNeXt + RBF SVM
classifier itself (93.3% / 97.1% / 98.1% hazard recall, half the parameters of
the hybrid). Its trade-offs: no dropout, so no MC-Dropout uncertainty (the
conformal sets would use the SVM's scores instead), and it is not the hybrid
architecture described in the project documents.

**Backbone ablation (Swin instead of ViT).** ConvNeXt + Swin-Tiny, 2 epochs on
the local GPU (55 min per epoch): 83.0% item / 88.6% family / 77.1% hazard
recall. Not a fair comparison with the 15-epoch hybrid — different epoch
budget and a different split — so it shows only that the backbone slot is
swappable. The measured fusion attention (ViT gets 7–18%) suggests a
ConvNeXt-only model is worth testing as the next ablation.

## Uncertainty and risk-bounded hazard routing

**MC-Dropout uncertainty.** 25 stochastic forward passes with dropout active
(normalisation layers frozen); uncertainty = entropy of the mean distribution /
log(33), in [0, 1]. A confident hazard is diverted immediately; an *uncertain*
hazard goes to priority review and is never auto-routed — on an out-of-distribution
test clip this cut false hazardous actuations from 12 to 0 (240 frames).

**What the simulations found about it** ([Simulations](#simulations)): as an error
detector it is *not better than the plain maximum class probability* (AUROC
0.598 vs 0.657 for misclassified items), and the fixed review
threshold of 0.5 sends 34.2% of all items to review, because label-smoothed training raises
the entropy of correct predictions. The two-tier gate structure is what helps;
its threshold should be calibrated to a review budget, not fixed at 0.5.

**Conformal family-set routing** (`decision_engine/conformal_routing.py`).
Top-1 routing lets a hazard reach a recycling bin whenever its single most
likely class is non-hazardous. Instead, item probabilities are summed per
family, a threshold per family is calibrated on held-out data (a stricter
α_H for hazardous), and routing follows the resulting *set* of plausible
families: any set containing "hazardous" is barred from non-hazardous bins;
automatic routing needs a single-family set. This gives a finite-sample bound
P(hazard reaches a non-hazard bin) ≤ α_H; a Beta-distribution rank correction
makes it hold for the specific calibration with probability ≥ 1 − δ (verified
on synthetic data: 2.1% of calibrations miss the target vs 39.5% without it).

Measured on the 89.30% model's verified test split (4,232 items, 738 hazardous),
300 random calibration/evaluation halvings per setting, non-hazard level α = 0.10;
baseline = max-softmax gate tuned to the **same** human-review workload. Top-1 routing
with no gate leaks 6.5% of hazards.

| α_H (hazard level) | Hazard leak, conformal (mean / 95th pct) | Items to review | Matched confidence gate (mean / 95th pct) | Advantage |
|---|---|---|---|---|
| 0.05 | **2.84% / 4.26%** | 5.7% | 3.50% / 4.58% | 1.2x |
| 0.02 | **1.57% / 2.90%** | 7.8% | 3.55% / 5.09% | 2.3x |
| 0.01 | **0.78% / 2.05%** | 31.1% | 1.77% / 4.52% | 2.3x |
| 0.005 | **0.29% / 1.11%** | 39.5% | 1.37% / 2.17% | 4.7x |

The measured leakage stays below α_H for every setting with α_H ≤ 0.05; the
advantage over a tuned confidence gate is 1.2–2.3x at practical workloads (about
8% review) and the mechanism's distinguishing feature is the *stated bound*. The
per-deployment (δ) variant is too conservative at this calibration size (about 11%
of items automatic). An earlier evaluation on a weaker 2-epoch checkpoint
(22.9% leak reduced to 0.54%) is superseded.

![conformal operating curve](reports/simulations/figures/fig_conformal_curve.png)

## Simulations

`python scripts/run_simulations.py` regenerates everything in this section
(`reports/simulations/summary.json` and `figures/`). Real-data simulations use the
89.30% model on its verified test split; sensor simulations use synthetic data and are
labelled as such. They were run to test the claimed mechanisms, and
several results are unfavourable — they are reported as found.

**Routing outcomes per mechanism** (200 halvings, gates calibrated to a 10% review budget):

| Configuration | Review share | Hazard leak | Non-hazard sent to hazard bin |
|---|---|---|---|
| top-1 class only | 0.0% | 6.60% | 0.96% |
| max-softmax gate | 10.1% | 3.30% | 0.68% |
| MC-dropout gate | 10.1% | 4.88% | 0.84% |
| conformal family sets | 5.6% | 2.88% | 1.14% |
| conformal + MC-dropout gate | 13.0% | 2.64% | 0.85% |
| conformal + max-softmax gate | 12.4% | **2.27%** | 0.66% |

![routing outcomes](reports/simulations/figures/fig_routing_ablation.png)

**Uncertainty estimator.** MC-dropout entropy: AUROC 0.598 for misclassified items vs 0.657 for
max-softmax; accepting the 50% most certain items gives 90.9% vs 93.2% item accuracy.

![uncertainty](reports/simulations/figures/fig_uncertainty.png)

**Distribution shift** (1,500 test images corrupted by noise, blur, darkness, occlusion at three
severities; thresholds calibrated on clean images only). The conformal bound is a statement about
inputs like the calibration set and does **not** survive corruption: at noise sigma 0.2 (item accuracy
53.9%) hazard leakage is 28.5% for top-1, 15.6% for conformal alone (target 5%), 9.8% with
a max-softmax gate added, 10.9% with the MC-dropout gate. Gates help (most on false hazardous routing, e.g.
blur sigma 4: 40.4% -> 12.5%) but do not restore the bound at severe corruption, and the max-softmax gate
flags more shifted images than MC-dropout.

![distribution shift](reports/simulations/figures/fig_distribution_shift.png)

**Contamination index with a lost sensor channel (SYNTHETIC sensors).** For linear models,
imputing a constant and re-tuning the threshold gives the *same decisions* as the dedicated
single-channel model (identical AUROC). The dedicated models' benefit is a calibrated score and a
sensitivity guarantee that holds in each availability case without recalibration: with the combined
model's threshold left in place, imputing zero for a lost gas channel drops sensitivity to
53.7% (target 95%), the dedicated model keeps 94.9%.

![oci dropout](reports/simulations/figures/fig_oci_dropout_synthetic.png)

**Training history of the improved recipe** (v2, Colab):

![v2 training history](reports/simulations/figures/fig_training_history_v2.png)

## Object detection

YOLO26n (2.4M parameters) fine-tuned on the TACO litter dataset, 640 px,
RTX 2050:

| Variant | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 1-class "waste item" localiser | 0.871 | 0.620 | **0.700** | 0.492 |
| 18-class TACO object identifier | 0.603 | 0.361 | 0.383 | 0.289 |

Naming the object as well as finding it costs 0.32 mAP50. Strong on rigid
objects (bottle cap 0.64, cup 0.61, can 0.59 mAP50), weak on tiny or
amorphous ones (pop tab 0.08, broken glass 0.06). 4.2 ms per image on the GPU.
A COCO-pretrained YOLO also names everyday objects (bottle, cup, banana) as a
prior over the material classifier — the hazard-exempt object-identity prior:
it can narrow a material guess, never suppress a hazard.

## Contamination index (OCI)

A greasy pizza box is cardboard but not recyclable; the camera cannot see
grease, so two sensors can: moisture and gas (MQ-135-type).

- Each reading is normalised against fixed calibration anchors to [0, 1].
- Three logistic regressions are fitted: both sensors, moisture only, gas only.
  If a sensor fails, the matching single-sensor model is used — no imputation.
- The operating threshold is the one with the lowest false-positive rate among
  thresholds reaching ≥ 95% sensitivity (a missed contamination costs more
  than a false reject).
- On **synthetic** calibration data (80 samples, 5-fold CV): combined AUC
  0.822, moisture-only 0.813, gas-only 0.732; the 95%-sensitivity threshold
  gives 97.9% sensitivity at 84% false-positive rate — the synthetic sensors
  are weakly separable, so real calibration data is essential.
- SHAP (exact for a linear model) explains each score per sensor; additivity
  error below 1e-6.

## Decision engine

Per item: hazardous and confident → hazardous bin; hazardous and uncertain (or
hazard in the conformal set) → priority manual review, never recycled;
non-hazard with uncertainty ≥ 0.5 → manual review; OCI above threshold on a
non-organic item → contamination reject; otherwise → the family's bin. The
actuator is a logged mock until hardware exists.

## Other components

| Component | What it does | Evidence |
|---|---|---|
| Grad-CAM (`explainability/gradcam_explanation.py`) | heatmap of image regions behind a prediction (ConvNeXt branch) | agrees with LIME on the same test image |
| LIME (`explainability/lime_explanation.py`) | superpixel perturbation explanation, model-agnostic | |
| SHAP (`contamination/shap_explanation.py`) | per-sensor contribution to the contamination score | additivity error < 1e-6 |
| SAM segmentation (`detection/sam_segmentation.py`) | box-prompted MobileSAM mask removes background before classification | 36% of a test box was background, excluded |
| DCGAN augmentation (`augmentation/gan_augmentation.py`) | per-class generator for rare classes | implemented; image quality (FID) not yet measured |
| Federated learning (`federated/fedavg_simulation.py`) | FedAvg across simulated sorting units, IID and non-IID shards; only weights leave a unit | single-process simulation |
| Video litter survey (`applications/video_litter_survey.py`) | ByteTrack IDs so each physical item is counted once | 160 per-frame detections merged into 16 inventory entries (240-frame test clip of 12 litter photographs; no ground-truth count was set) |
| ONNX export (`classification/export_onnx.py`) | edge-runtime model with PyTorch-parity check | max abs difference 9.42e-6; 203 MB classifier, 9.3 MB detector |

## Model cost

Measured by `src/edgewaste/analysis/model_complexity.py` (batch 1, 224x224):

| Component | Parameters |
|---|---|
| ConvNeXt-Tiny backbone | 27.82 M |
| ViT-Small/16 backbone | 21.67 M |
| Projections | 0.59 M |
| Attention fusion | 0.13 M |
| Classifier head | 0.54 M |
| **Total** | **50.75 M** |

| Cost | Value |
|---|---|
| Compute | 17.4 GFLOPs per image (8.7 G multiply-adds) |
| Latency, RTX 2050 laptop GPU | 14.0 ms |
| Latency, laptop CPU | 115.5 ms |
| Checkpoint / ONNX file | 203 MB / 203 MB (fp32) |
| Detector (YOLO26n) | 2.4 M parameters, 4.2 ms, 9.3 MB ONNX |

Both are inside the < 500 ms per item target on a laptop; Raspberry Pi
latency is unmeasured. fp16 or int8 quantisation would roughly halve or
quarter the file size (not yet done).

## Reproducibility

```bash
python scripts/reconstruct_colab_split.py                        # exact Colab split -> data/splits_colab_reconstructed.csv
python scripts/collect_analysis_features.py --expect-acc 0.8930 # caches; fails fast if the split is wrong
python scripts/run_ml_analysis.py                                # reports/ml_analysis (metrics, fit, calibration,
                                                                 # imbalance, baselines, projections, cost)
python scripts/evaluate_conformal_routing.py --config configs/classifier_convnext_vit.yaml --ckpt runs/stage1/best.pt
python scripts/run_simulations.py                                # reports/simulations (uncertainty, conformal, shift, routing, OCI dropout)
```

Seeds are fixed (42 for data and training, 0 for analysis resampling);
analysis inference runs in fp32 because fp16 flipped one of 4,232 predictions.

### Colab troubleshooting

- **Steps per epoch must be 617** (19,738 training images / 32). Double that means stale
  duplicates in `data/processed` on Drive; the Step 6 gate now stops this. Cure: delete
  `data/processed` and `data/splits.csv` on Drive (keep `data/downloads`), re-run Step 6.
- **Step 8 must print `backbones frozen (linear-probe phase)` for epochs 1–2**, otherwise Colab is on old code.

## Limitations

- **No physical hardware yet** — sensors are simulated, the contamination
  model is fitted on synthetic data (weakly separable: 84% false positives at
  95% sensitivity), the actuator is a log line, and on-device latency is
  unmeasured on a Raspberry Pi.
- **Domain shift:** training images are studio and curated household photos.
  On outdoor litter footage and on corrupted images accuracy drops sharply; the
  uncertainty and conformal gates reduce but do not remove the hazard risk, and the
  conformal bound does not hold under shift (see Simulations).
- **The MC-dropout uncertainty is not better than max-softmax** at flagging errors in
  this model (AUROC 0.60 vs 0.66), and its fixed 0.5 threshold over-flags (34% of items).
- **The fine-tuning recipe is sub-optimal** — see [Improving the classifier](#improving-the-classifier).
- **Look-alike classes** (cardboard boxes vs packaging, steel vs aluminium
  food cans) stay near 0.6 F1; they share a bin, so routing is unaffected.
- The main run's per-epoch history is on Google Drive, not in this repository.

## Review FAQ

**Which optimiser?** AdamW, peak LR 3e-4, weight decay 0.05, OneCycle cosine
schedule with 1-epoch warm-up, gradient clipping at 1.0. Adaptive steps suit
fine-tuning two pretrained backbones plus a new head; decoupled weight decay
is the correct form of L2 regularisation under Adam.

**Which activation functions?** GELU in all hidden layers, softmax for fusion
attention and outputs. See [Model architecture](#model-architecture).

**Which loss?** Cross-entropy with label smoothing 0.1.

**Is it overfitting or underfitting?** Neither harmfully: 96.9% train vs
89.3% test with validation still improving at the last epoch, and val ≈ test.
See [Overfitting and underfitting](#overfitting-and-underfitting).

**Is class imbalance handled?** Yes — and the analysis found it was handled
twice, which over-corrects slightly; measured, explained and fixed. See
[Class imbalance](#class-imbalance).

**Were PCA / LDA / SVM used?** Yes, as baselines and analysis tools. See
[Classical ML baselines](#classical-ml-baselines).

**Why deep learning at all?** The best classical pipeline on hand-crafted features (HOG + colour histogram → PCA → RBF SVM) reaches 71.1%; deep features reach 89–93% (McNemar p ≈ 5e-135). Classical classifiers stay useful *on top of* deep features: an RBF SVM on frozen ImageNet ConvNeXt features is currently the most accurate classifier in the project (93.3%). See [Classical ML baselines](#classical-ml-baselines).

**Why a hybrid of two backbones?** See [Model architecture](#model-architecture)
and the measured [backbone attention](#what-the-model-learned).

**How do you know the test set is clean?** Stratified seeded split, val-only
model selection, a leakage bug found and fixed, and the Colab split recovered
and verified number-for-number. See [Splits](#splits-and-leakage-prevention).

**Why no k-fold cross-validation for the deep model?** One training run is
about an hour of GPU time; the 95% bootstrap intervals on 4,232 test images
already quantify the uncertainty (accuracy ± ~0.9 points). The small
contamination model does use 5-fold CV.

**Are the probabilities trustworthy?** After temperature scaling, largely yes
(ECE 7.3%). As trained the model is under-confident. See [Calibration](#calibration).

**How are uncertain predictions handled?** MC-Dropout entropy gate plus
conformal family sets; see [Uncertainty](#uncertainty-and-risk-bounded-hazard-routing).

**Which metrics and why so many?** Accuracy hides minority classes, so the
report adds balanced accuracy, macro F1, kappa, MCC, AUCs, calibration error,
and the two operational metrics: family (routing) accuracy and hazard recall.

## Video and live-camera sorting (segmentation + segregation)

`edgewaste-video-sorter` (`src/edgewaste/applications/video_sorter.py`) takes a video file, a folder of frames or a live camera (`--source 0`). For every item it draws the segmentation outline, tracks it across frames (ByteTrack), classifies the mask-cropped item with the 33-class model, and chooses the bin. Outputs: `annotated.mp4`, `detections.csv`, `inventory.csv`, `summary.txt`.

- **Dataset:** ZeroWaste-f (Bashkirova et al., 2022, arXiv 2106.02740, Zenodo record 6412647): real conveyor-belt video frames with polygon masks for rigid plastic, cardboard, metal and soft plastic. `scripts/prepare_zerowaste_video.py` streams it (the zip is 7.5 GB, never stored) into `data/video/zerowaste/` (git-ignored). Used here: 1,001 train (every 3rd frame, shuffled), 305 val (every 2nd), 255 test frames (videos 05, 08, 09 only; the full test split was not downloaded because the connection was slow).
- **Model:** YOLO26n-seg fine-tuned by `scripts/train_video_segmenter.py` (38 epochs, early stopped, batch 8, 640 px, RTX 2050) -> `runs/video_seg/zerowaste_seg_a/weights/best.pt`.
- **Decision:** the video-trained segmenter decides the material when its voted confidence is at least 0.40; the 33-class model gives the fine class. Safety rule kept: if the calibrated conformal set contains hazardous AND the classifier's hazardous probability is at least `--haz-mass` (0.90), recycling bins are barred. `startup_calibration.py` sets the uncertainty threshold and conformal thresholds from the validation cache at start-up.
- **Evaluation** (`scripts/evaluate_video_sorter.py` -> `reports/video/evaluation.md`, 255 test frames, 470 detections; mask-IoU >= 0.5 matching):

| quantity | value |
|---|---|
| box mAP50 / mAP50-95 | 0.325 / 0.232 |
| mask mAP50 / mAP50-95 | 0.330 / 0.226 |
| per-class mask mAP50 | rigid plastic 0.34, cardboard 0.46, metal 0.003, soft plastic 0.52 |
| ground-truth items found | 32.3% |
| family accuracy of matched items: segmenter only / 33-class model only / fused | 91.8% / 2.0% / 76.1% |
| false hazardous-bin routes (dataset has no hazardous items) | 59 of 470 |
| speed | about 13.5 frames per second (segment + track + classify) |

- **Honest reading:** segmentation is modest (a nano model, 1,001 frames, many small overlapping items; metal has very few examples and is essentially not learned). The photo-trained 33-class model is badly out of domain on conveyor crops (2.0% family accuracy and it frequently puts hazardous in the set), which is why the video-trained segmenter decides the material. With the hazard gate at 0.50 the fused accuracy was 45.5% (204 false hazard routes); at 0.90 it is 76.1% (59). Fused stays below segmenter-only because those 59 safety overrides are kept. Tracks are short (about 2 frames) because frames are 10 apart. Test video 09 has training frames as close as 60 frames, so its numbers are optimistic. Not tested on a live camera yet.
- **Not done:** hazardous items are not in ZeroWaste, so hazard recognition on video comes only from the photo-trained classifier and is unvalidated on video.

## Hardware prototype (two-day, minimum cost)

A bench unit with a camera, a moisture sensor, an MQ-135 gas sensor, one servo that tilts a tray, three leds and a buzzer, driven by an Arduino over USB serial; the laptop runs the models. Full plan (parts about ₹1,000-1,800, wiring, build, calibration, schedule, tests): [docs/hardware_prototype_plan.md](docs/hardware_prototype_plan.md). With an ESP32, a DHT22 (humidity replaces the moisture probe), an MQ-135 and a servo, follow [docs/esp32_prototype_walkthrough.md](docs/esp32_prototype_walkthrough.md) and use `scripts/check_node.py` for bring-up.

```bash
pip install -e ".[hardware]"                                         # pyserial
python scripts/prototype_sorter.py --mock-node --image <photo>       # whole software path with no hardware
python scripts/calibrate_prototype_oci.py collect --port COM5        # then:  ... fit   (real contamination calibration)
python scripts/prototype_sorter.py --port COM5 --camera 0            # live: button or SPACE classifies one item
```

Status: the software path is tested end to end in mock mode (image to decision to printed action, calibration collect and fit on invented data); the firmware and the real serial link have not been run on hardware yet. Calibration uses the validation split of the verified first model (`runs/stage1`), never the test split. Decisions are logged to `runs/prototype/log.csv`.

## Project report and review slides

The university report (template `Project-1 Report final.docx`: A4, left margin 1.5 in, Times New Roman 12, 1.5 spacing, chapter headings 14 pt capitals, roman then arabic page numbers, Word equations numbered by chapter, APA references, nine chapters plus Appendix A) and the 25-slide review deck are built by scripts, so every number comes from the repository:

```bash
python scripts/report/report_figures.py              # extra figures -> docs/report/figures
# diagrams: mmdc -i docs/report/mermaid/<name>.mmd -o docs/report/figures/diag_<name>.png -s 2 -b white
python scripts/report/make_report.py                 # raw docx (text lives in scripts/report/report_ch1..8.py)
powershell scripts/report/finalize_report.ps1        # Word builds contents/figure/table lists, saves docx + pdf
python scripts/report/build_slides.py                # docs/report/BITE497J_Project_I_Review_Slides.pptx (2 embedded videos)
```

Outputs: `docs/report/BITE497J_Project_I_Report.docx` (115 pages: front matter i–xix, chapters 1–9 on pages 1–90, appendix without page numbers) and `BITE497J_Project_I_Review_Slides.pptx`. References are in `scripts/report/report_refs.py` (APA 7). After editing the docx by hand, right-click the contents, figure and table lists and choose Update Field.

## Changelog

| Date | Change |
|---|---|
| 2026-10-09 | Video and live-camera sorting: ZeroWaste-f conveyor-video dataset (streamed, git-ignored), YOLO26n-seg trained on it, `edgewaste-video-sorter` (segment + track + classify + route), `startup_calibration.py`, `scripts/evaluate_video_sorter.py`. Test (255 frames, 3 videos): mask mAP50 0.33; family accuracy 91.8% segmenter-only, 2.0% photo-trained classifier-only, 76.1% fused; 59/470 false hazard routes after adding a 0.90 hazard-probability gate (0.50 gave 204). Live camera untested. |
| 2026-10-05 | ESP32 variant of the prototype: `hardware/esp32/sorter_node_esp32/sorter_node_esp32.ino` (DHT22 humidity x10 as the moisture channel, MQ-135 through a voltage divider, servo, BOOT-button trigger), `docs/esp32_prototype_walkthrough.md`, `scripts/check_node.py` bring-up tool, and a more robust serial link that skips ESP32 boot text. Serial class tested against a fake ESP32; firmware not compiled or run on hardware. |
| 2026-10-03 | Two-day hardware prototype plan and software: `docs/hardware_prototype_plan.md` (bill of materials about ₹1,000-1,800, wiring, mechanics, hour-by-hour schedule, tests), Arduino firmware `hardware/arduino/sorter_node/sorter_node.ino`, `scripts/prototype_sorter.py` (camera + sensors + decision engine + tray commands, with `--mock-node`), `scripts/calibrate_prototype_oci.py` (collect and fit real contamination calibration), `edgewaste.applications.prototype_node` (serial link and mock). Tested in mock mode on real images and on invented sensor data; not yet run on physical hardware. |
| 2026-10-02 | References from the Review 1 submission were re-checked against Crossref; corrections: Alkılınç et al. year and author list, Arun issue number, Radchenko and Fill identified (arXiv 2403.09141). Hardware cost figures are planning estimates. |
| 2026-10-02 | Closest earlier work on camera + sensors is Chu et al. 2018. v12 (51-paper reference list) was superseded as too long. Plain list: `docs/references.md`. |
| 2026-10-01 | Demonstration gallery (`scripts/make_demo_gallery.py` -> `reports/demo/gallery/`, run one part at a time: `classes`, `hazards`, `gradcam`, `detector`, `videos`, `oci`, `federated`, `gan`): all 33 classes with routing on random held-out images, per-class F1 chart, hazardous-family gallery, Grad-CAM on 16 classes, 18-class detector and MobileSAM cut-outs on benchmark photos, `class_tour.mp4`/`.gif` (33-class tour with routing banner), survey filmstrip/inventory charts/`survey.gif`, Organic Contamination Index response on simulated sensors, a new FedAvg simulation on stored embeddings (5 non-IID units, 15 rounds: 89.4% vs 89.0% central vs 81.4% mean local; backbone frozen, head averaged), GAN sample grid. A simulated 60-item conveyor figure was generated but left out of the draft because at the 95%-sensitivity threshold it flags most clean items too (see Table 7), which makes it a poor demonstration. |
| 2026-10-01 | Demonstration snapshots (`scripts/make_demo_snapshots.py` -> `reports/demo/`: routing sheet on eight held-out images, explanation panel, annotated video frames, `worked_example.json`). **Correction:** the earlier claim "19 unique items, matching the true count" for the video-survey test clip could not be reproduced (re-runs: 16 with the 18-class detector, 13-17 with single-class detectors) and the clip (12 litter photographs x 20 frames) has no ground-truth item count; now reported as 160 per-frame detections merged into 16 inventory entries. |
| 2026-10-01 | Figure labels in the simulation and analysis scripts now use spelled-out terms (Monte Carlo dropout, maximum-probability gate, principal component analysis, expected calibration error, and so on) and all figures were regenerated; numbers are unchanged. Simulated-sensor wording replaces "synthetic" in figure titles. |
| 2026-10-01 | Simulation suite (`scripts/run_simulations.py`) testing each claimed mechanism: uncertainty-estimator quality, conformal operating curve and bound validity, routing outcomes per mechanism, behaviour under image corruption, and sensor-dropout handling on synthetic data. Findings reported as found: conformal bound valid for in-distribution inputs (2.3x lower leakage than a tuned confidence gate at ~8% review) but not under shift; MC-dropout is not a better error detector than max-softmax and its fixed 0.5 threshold over-flags; for linear models, dedicated single-channel OCI models match imputation with a re-tuned threshold. Synthetic generator gained `noise_scale`/`anchors` options (defaults unchanged). |
| 2026-10-01 | v2 retrain finished (Colab, 12 epochs, 48 min): **93.83% item / 97.28% family / 98.37% hazard recall** (Colab-reported; validation 93.86% at epoch 8) vs 89.30 / 94.14 / 93.50 for v1 — the fine-tuning fix worked. A local re-score gave an inflated 96.8% because the local split differs from Colab's; discarded and quarantined (`LEAKY_local_split_*`). `edgewaste-eval` now refuses a manifest whose validation accuracy does not match the one recorded in the checkpoint. |
| 2026-09-30 | Caught a duplicate-data hazard during the first Colab retrain (1,234 steps per epoch = 2x the 617 expected): the Drive `data/processed` folder still held the earlier run's old-named copies of every photo, so the split contained each image twice and could put one photo in both train and test. `make_data_splits` now splits only the images listed in the latest `provenance.csv` (ignoring and reporting stale files); the Colab notebook's Step 6 gate asserts split size == provenance size. The affected run was stopped and restarted on clean data. |
| 2026-09-30 | Full ML analysis suite (`src/edgewaste/analysis/`, `scripts/run_ml_analysis.py`): complete metrics with bootstrap CIs, fit diagnostics, calibration, class imbalance, classical baselines (PCA/LDA/SVM/kNN/LR/RF/NB), embedding projections, backbone attention, model cost. Recovered and verified the exact Colab split. Found and fixed the double class-imbalance correction (`imbalance_strategy`). Added discriminative backbone learning rate (`backbone_lr_mult`) and moved the main config to an improved recipe writing to `runs/stage1_v2`. Conformal routing re-evaluated on the main model. `edgewaste-eval --manifest`. This readme rewritten as the universal reference. |
| 2026-09-30 | Repository restructured into meaningful lowercase folders and file names; one-line header comment in every file. |
| 2026-09-25 | Conformal hazard-leakage bound added (`conformal_routing.py`). |
| 2026-09-23 | 33-class classifier trained on Colab (89.30%); corrupt-image and leakage bugs fixed; hazard confidence floor; ONNX export. |
| 2026-09-22 | Taxonomy expanded from 7 to 33 classes / 9 families; 18-class detector; video survey; object-identity prior. |
| 2026-07 | Stage 1 pipeline, 7-class classifier, 1-class detector. |
