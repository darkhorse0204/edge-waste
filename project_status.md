<!-- project_status.md - current status: real trained results, bugs found and fixed, open risks -->

# project_status.md
## Edge-Deployed Waste Classification with Sensor-Augmented Vision

*Owner: Ansh | Last updated: 2026-09-23*

---

## 📍 Current Phase

**Stage 1 vision pipeline is trained, evaluated, and verified end-to-end on
real checkpoints — both the classifier and both detector variants.** Stage
2/3 (OCI, confidence, decision engine, federated learning, explainability,
video survey) is built and integration-tested in software, with physical
hardware (sensors, conveyor, Raspberry Pi) simulated — see readme.md's
architecture tables for the full breakdown.

Faculty feedback (received 2026-09-22, after the classifier had already
started once on the old taxonomy) required expanding the class count from
7 to >15–20. The taxonomy was redesigned to **33 fine-grained item classes
in 9 material families**, and retrained — the numbers below are from that
run, not the original 7-class one.

---

## ✅ Trained models — real, verified numbers

### Classifier — 33-class ConvNeXt+ViT hybrid (`runs/stage1/best.pt`)

Trained on Colab (T4 GPU), 15 epochs, 28,202 images across 33 classes in
9 material families. Evaluated on a 4,232-image held-out test split.

| Metric | Result |
|---|---|
| **Item-level accuracy** (33 classes) | **89.30%** |
| **Family-level accuracy** (9 families, i.e. routing correctness) | **94.14%** |
| **Hazardous recall** (battery/e-waste/medical) | **93.50%** (690/738) |
| Macro F1 (item level) | 0.8626 |

**Why report two accuracy numbers.** A flat classification metric hides the
question a physical sorter actually cares about — not "did it name the
exact item" but "did it end up in the right bin." The 4.84-point gap between
item (89.30%) and family (94.14%) accuracy tells that story directly: most
confusion is between near-identical items *within* the same family
(`aluminum_food_cans` ↔ `steel_food_cans`, both 0.62 F1, both route to
`metal`; `cardboard_boxes` ↔ `cardboard_packaging`, both route to
`cardboard`) — a labelling error with zero sorting consequence. See
`edgewaste-eval`'s output (`classification/evaluate_classifier.py`) for the full per-class breakdown
and both confusion matrices.

**Known limitation, stated plainly:** the training corpus is studio and
curated real-world household photography. Tested against genuinely
out-of-distribution input (TACO outdoor litter crops via the video survey
pipeline), the classifier's raw predictions degrade sharply and the
MC-Dropout uncertainty estimator correctly flags it (mean uncertainty 0.68
vs. near-zero on in-distribution images) — the confidence gate is doing its
job, but a real beach-survey deployment would need either TACO-domain
fine-tuning or accepting a much higher manual-review rate on field footage.
This is future work, not a defect to be hidden.

### Detector — two trained variants, kept both deliberately

| Model | Precision | Recall | mAP50 | mAP50-95 | Params |
|---|---|---|---|---|---|
| **1-class** `waste_item` localiser | 0.871 | 0.620 | **0.700** | 0.492 | 2.375M |
| **18-class** TACO litter identifier | 0.603 | 0.361 | **0.383** | 0.289 | 2.375M |

Both are YOLO26n, trained locally (RTX 2050, 4GB) on the TACO Kaggle mirror.
Keeping both gives a real ablation rather than an assertion: asking one
model to also *name* the litter object (not just localise it) costs 0.32 in
mAP50 and drops recall from 0.62 to 0.36.

Per-class detector results split exactly where theory predicts: strong on
rigid, visually consistent objects (Bottle cap 0.639, Cup 0.612, Can 0.593),
weak on the tiny (Pop tab 0.079), the amorphous (Broken glass 0.064,
Cigarette 0.177), and the two literal catch-alls (`Other litter`,
`Unlabeled litter`) that have no consistent appearance by definition.
Inference speed 4.2ms/image — well inside the <500ms/item edge target
(though that target is against a Tesla T4/RTX 2050, not a Raspberry Pi;
on-Pi benchmarking is still unrun, see Known Risks).

### ONNX export

Both models exported with PyTorch-parity verification (`classification/export_onnx.py`):

- Classifier: `runs/stage1/model.onnx` (203.2 MB) — max abs diff vs PyTorch
  **9.42e-06** (atol 1e-3)
- Detector (18-class): `runs/detect/model.onnx` (9.3 MB) — verified by direct
  inference comparison on a real image: identical detection count, identical
  confidence to 3 decimal places (0.778 vs 0.778)

---

## 🏗️ Architecture — what actually got built

### Two-level taxonomy, cross-validated

| Level | Model | Classes |
|---|---|---|
| What object is it? | YOLO26n (18-class variant) | 18 TACO litter categories |
| What is it made of? | ConvNeXt+ViT hybrid | 33 items → 9 material families |
| How contaminated? | OCI (moisture+gas sensor fusion) | continuous 0–1 |
| How sure are we? | MC-Dropout | normalised predictive entropy |

**Object-Identity Prior** (`decision_engine/object_identity_prior.py`) cross-checks the two vision
models against each other: a crop the detector confidently calls a `Can`
has its material distribution re-weighted toward `metal`. The prior is
*soft* — a confident contradictory classifier call still survives and gets
flagged for manual review rather than silently overwritten. Hazardous
material calls are exempt from down-weighting entirely: the prior must
never be able to talk the classifier out of flagging a hazard.

### Decision engine, with a hazard confidence floor added after a real failure

`decision_engine/routing_rules.py` routes each item by material *family* (a water bottle and a
soda bottle share a chute). Hazard calls bypass the normal uncertainty gate
and act immediately — **but only when confident**. Running the *real*
trained classifier against out-of-distribution video crops surfaced that an
uncertain classifier disproportionately guesses hazard classes on unfamiliar
input, which would have fired 12 unconditional hazardous-stream actuations
on a single 240-frame test clip for items that were physically cardboard and
cups. Fixed: an uncertain hazard call now routes to a flagged priority
manual review instead of automatic actuation, while the core safety
guarantee (no hazard call ever reaches a recycling/compost gate,
confident or not) is preserved unconditionally. Verified before/after on the
same test clip: false hazardous actuations 12 → 0.

### Video survey pipeline (`edgewaste-video`)

Processes video of multi-item scenes (a beach survey walk, a conveyor run)
and produces a de-duplicated litter inventory. The hard part is counting,
not detection: one physical item spans hundreds of frames, so naive
per-frame summing overcounts by orders of magnitude. Each detection carries
a persistent ByteTrack ID; the inventory aggregates per track. Verified on a
synthetic survey video: 19 unique items correctly tracked across 240 frames
(vs. hundreds of raw per-frame detections).

### Federated learning simulation (`federated/fedavg_simulation.py`)

Single-process FedAvg across virtual clients (IID and non-IID shard
partitioning), verified end-to-end on this machine's GPU. Real distributed
deployment across physical devices is documented as a scoped next step, not
built — a single-process simulation already demonstrates the FedAvg
mechanism; multiple physical devices would demonstrate transport, not
different math.

### Everything else built, hardware simulated

`contamination/simulated_sensors.py` (simulated moisture/gas/metal/load-cell, per-family-plausible),
`contamination/` (fixed-reference normalisation, 3-model logistic fit, 5-fold CV
ablation, sensitivity-constrained threshold — fitted on synthetic
calibration data pending the real 160-sample physical protocol),
`decision_engine/mc_dropout_uncertainty.py` (MC-Dropout), `explainability/gradcam_explanation.py` (Grad-CAM, ConvNeXt branch),
`applications/prediction_logger.py` (per-detection CSV), `scripts/run_live_camera.py` /
`edgewaste-pipeline` (live detect→classify→OCI→decision loop).

---

## 🐛 Real bugs found by actually running the system, and fixed

Not hypothetical robustness — every one of these was reproduced with the
real trained checkpoint or the real training run, not caught in review.

1. **Corrupt JPEG killed the first Colab training attempt at epoch 5**
   (TrashBox's e-waste folder, `UnidentifiedImageError`, 4 finished epochs
   lost). Fixed in `dataset.py` (substitutes a neighbouring sample, reports
   each bad path once, gives up only after 50 consecutive failures) and
   `ingest_raw_datasets.py` (rejects undecodable files before they reach the manifest at
   all). Verified: the retrained run hit the same file repeatedly and kept
   going.
2. **Cross-machine data leakage in the train/test split.** `_dedupe_name`
   hashed the *absolute* source path, so the same downloaded file produced a
   different generated filename on Windows vs. Colab's Linux runtime —
   different filename → different alphabetical sort order → `train_test_split`
   (same seed, differently-ordered input) assigned the same image to a
   different split on each machine. Discovered because a local re-evaluation
   of the Colab-trained checkpoint scored 94.52%, five points *above*
   Colab's own honest held-out result (89.30%) — a re-run scoring higher is
   the signature of leakage, not improvement. Fixed by hashing the path
   relative to the source root, POSIX-normalised. **Colab's originally
   reported 89.30%/94.14%/93.50% remain the only valid numbers for this
   checkpoint** — the fix matters for every run after this one.
   *Update 2026-09-30:* the exact Colab split was recovered
   (`scripts/reconstruct_colab_split.py`) and verified number-for-number
   (3779/4232 = 89.30%, 94.14%, 690/738), so the checkpoint can now be
   analysed locally without leakage.
6. **Class imbalance was corrected twice** (balanced sampler + inverse-frequency
   loss weights under one flag), over-weighting rare classes ~1/n². Measured
   with post-hoc logit adjustment: small effect (+0.3–0.6 pt accuracy, +1.2 pt
   hazard recall). Fixed: `train.imbalance_strategy`, default `sampler`.
3. **Hazard routing had no confidence floor** — see Decision engine above.
4. **Colab's `git pull` could silently leave the session on old code** if the
   Drive checkout had diverged — fixed with a hard-reset-to-origin step plus
   a taxonomy/pretrained-flag assertion gate that fails loudly instead.
5. **Colab clone into a non-empty, non-git Drive folder** (from an earlier
   aborted session) failed with a misleading "check your token" error —
   fixed to convert the folder into a checkout in place rather than deleting
   gitignored data.

---

## 🚧 Known Risks / Open Decisions

| Risk | Status | Notes |
|---|---|---|
| On-Pi latency | **Unverified** | 4.2ms/image detector inference is on RTX 2050/T4, not a Raspberry Pi 5. No hardware in hand. |
| OCI on real sensors | **Blocked** | Math validated on synthetic data only; needs ESP32 + moisture + MQ-135, mandatory 24–48h burn-in, 160 staged calibration samples. Nothing else in the project blocks starting this in parallel. |
| Classifier on out-of-distribution input | **Known limitation** | Degrades on genuinely field-like imagery (see classifier section above); MC-Dropout correctly flags it, but no domain-adaptation work done yet. |
| 1-class vs 18-class detector | **Both kept, not resolved** | Deliberately: gives a real granularity/accuracy ablation for the report rather than forcing a premature choice. |
| Fine-tuning recipe | **Improved config ready, retrain pending (Colab)** | See readme "Improving the classifier". Main config now: 25 epochs, one imbalance correction, backbone LR x0.1, output `runs/stage1_v2` (cannot overwrite the reported checkpoint). |
| SHAP (OCI) / LIME (classifier) | **Done** | `contamination/shap_explanation.py` (exact LinearExplainer, additivity error < 1e-6), `explainability/lime_explanation.py`; CLI: `scripts/demo_explainability.py`. |
| SAM segmentation | **Done (optional stage)** | `detection/sam_segmentation.py`, enabled with `edgewaste-video --segment`. |
| DCGAN augmentation | **Done** | `augmentation/gan_augmentation.py`, `scripts/train_gan_augmentation.py`, for genuinely imbalanced classes only. |
| Calibrated hazard-leakage bound (`decision_engine/conformal_routing.py`) | **Done** | Family-marginalised, family-conditional split-conformal sets with a Beta-corrected (PAC) rank; routing follows the set (`decide(..., family_set=...)`), any set containing `hazardous` is barred from non-hazard gates. Re-evaluated 2026-09-30 on the **main 89.30% model** (recovered split, 1,000 cal/eval halvings): top-1 leaks 6.50% of hazards; conformal at α_H=0.05 leaks 2.83% with 94.3% auto-routed (matched-workload threshold gate 3.48%); at α_H=0.02, 1.13% vs 1.40% with 58.6% auto. The δ=0.05 per-deployment version is impractical here (10.8% auto). Earlier Swin-checkpoint numbers (22.9% → 0.54%) are superseded for reporting. `scripts/evaluate_conformal_routing.py`; results in `runs/stage1_swin/conformal_results.json`. |
| Swin ablation (ConvNeXt+Swin vs ConvNeXt+ViT) | **Done, 2 epochs only** | ConvNeXt+Swin, 2 epochs on the RTX 2050 (~55 min/epoch): best val acc 0.834; **held-out test: 83% item / 88.56% family / 77.1% hazardous recall (569/738)**. **Not a valid head-to-head against the ViT hybrid (89.30% / 94.14% / 93.50%)**: that model trained 15 epochs on Colab, this one 2, with a 2-epoch LR schedule, and the two runs use different train/test splits (the split-hashing fix landed after the Colab run). Treat this as "Swin is trainable and learns quickly in this slot", not "Swin beats or loses to ViT". A fair comparison needs 15 epochs on the same split (Colab). Artifacts: `runs/stage1_swin/`. |
| Federated learning | Simulation only | Real physical-device deployment not attempted; documented as a scoped next step. |
| WaRP dataset | Available, not integrated | `parohod/warp-waste-recycling-plant-dataset` on Kaggle (845 MB, 28 classes, real plant conditions) — candidate for a robustness evaluation. |

---

## 🔧 Environment Status

| Component | Status (2026-09-23) |
|---|---|
| Local GPU | NVIDIA RTX 2050, 4GB VRAM, CUDA 12.1 — used for both detector trainings; too small for the 50.7M-param classifier in reasonable time |
| Classifier training | Google Colab, T4 GPU, via `notebooks/colab_classifier_training.ipynb` (private repo, GitHub-token-authenticated through Colab Secrets) |
| `torch` | 2.5.1+cu121 (local) |
| `ultralytics`, `onnx`, `onnxruntime`, `grad-cam`, `lap` | Installed |
| Disk space | Tight — 12GB free on C: at last check; datasets are ~4GB, checkpoints ~200MB each |

---

## 📅 Session Log

| Date | What was done |
|---|---|
| 2026-07-30/31 | Original 7-class Stage 1 build: 95.1% val acc classifier, 68.7% mAP50 1-class detector (on the machine later found to be a different one than this working directory). |
| 2026-09-22 | Found this working directory had regressed to "code complete, untrained." Rebuilt Stage 2/3 in software (OCI, confidence, decision engine, federated sim, ONNX export, explainability) with hardware simulated. Discovered a second machine's work (`detection/coco_object_recognizer.py`, `applications/prediction_stabilizer.py`, richer `applications/live_camera_pipeline.py`) had actually been pushed to GitHub and merged it in, catching a silent `pretrained: false` regression in the process. Kicked off Colab training. |
| 2026-09-22 (later) | Faculty required >15–20 classes. Researched real Kaggle datasets by API rather than search snippets; redesigned taxonomy to 33 items / 9 families. Discovered TACO's raw download already carried 18 real litter categories the old code discarded — trained an 18-class detector locally alongside the existing 1-class one. Added the Object-Identity Prior, the video survey pipeline with ByteTrack counting, and hierarchical (item + family level) evaluation. Diagnosed and fixed a private-repo Colab auth flow (GitHub token via Colab Secrets) and two Colab-specific failure modes (stale `git pull`, non-empty non-git Drive folder). |
| 2026-09-24 | Faculty round 2 (handwritten notes): implemented SHAP, LIME, SAM segmentation, DCGAN augmentation and a Swin ablation config. Drafted the patent Invention Disclosure Form (IDF, kept out of git deliberately: pre-filing disclosure risk). |
| 2026-09-30 | Restructured the repo (meaningful lowercase names, header comments). Built the ML analysis suite (`src/edgewaste/analysis/`, `scripts/run_ml_analysis.py` -> `reports/ml_analysis/`): full metrics + bootstrap CIs, fit diagnostics (train 96.89% / val 89.51% / test 89.30%, best epoch = last), calibration (under-confident, ECE 16.6% -> 7.3% with T=0.71), class imbalance (double correction found and fixed), classical baselines, embedding projections, fusion attention (~85% ConvNeXt), model cost. Recovered the exact Colab split. Discriminative LR + improved main config. readme.md rewritten as the universal reference. |
| 2026-09-23 | Classifier finished training on Colab: **89.30% item / 94.14% family / 93.50% hazardous recall**. Fixed a corrupt-JPEG crash (same failure class the original Aug audit had already flagged once) at both the dataset and ingest level. Downloaded the real checkpoint locally; a local re-evaluation scoring *higher* than Colab's own result exposed a cross-machine data-leakage bug in the split-generation hashing, fixed. Ran the video survey pipeline against the real checkpoint on out-of-distribution crops, which surfaced a hazard-routing confidence-floor gap (12 false hazardous actuations on one test clip) — fixed and verified (12 → 0). Exported both classifier and detector to ONNX with parity verification (classifier: 9.42e-06 max abs diff). |
