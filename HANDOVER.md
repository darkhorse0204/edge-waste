# Handover — Edge-Deployed Waste Classification

*Last updated: 2026-07-30. Written for whoever picks this up next — a
teammate, your future self, or your guide checking progress. Read this before
`README.md`; this is the "where things actually stand" document, README is
the command reference. This version folds in the granular execution detail
from `Master-Work-Plan.md` (repo root) — that document is the fullest,
most detailed protocol available for everything past vision, and is cited
throughout below rather than duplicated wholesale.*

---

## 0. Document map — don't get lost

| Document | What it's for | Authority |
|---|---|---|
| **`HANDOVER.md`** (this file) | Current status + what to do next, reconciling all the docs below against actual code state | You're reading it |
| `README.md` | Command reference for what's actually built | Current |
| `PLAN.md` | Decision log (taxonomy, pipeline architecture, superseded choices) | Current |
| `Master-Work-Plan.md` | **Full granular execution roadmap** — every phase from env setup to paper writing, with pin diagrams, code snippets, download links, verification criteria | Authoritative for anything not yet built, **with two flagged exceptions — see §2.4** |
| `Waste-Classification-Tech-Stack-Recommendation (3).md`, `Project-Action-Plan.md`, `OCI-Formula-Design-and-Calibration-Protocol.md` (parent folder, one level above this repo) | Original architecture/OCI design docs | Superseded in detail by `Master-Work-Plan.md`, but still the best source for *why* each design choice was made (novelty framing, literature grounding) |
| `docs/stage-{1,2,3}-*.md` | Older blueprint-derived checklists | Mostly superseded by `Master-Work-Plan.md`'s phase structure; kept for the blueprint's original module numbering if you need to cross-reference a guide's original brief |

---

## 1. What exists right now (code complete, nothing has been run)

Everything in this section is **code and config only** — no dataset has been
downloaded, no model has been trained, no sensor has been wired, and no
checkpoint exists anywhere in this repo (`data/` and `runs/` are git-ignored
and currently empty/absent). This maps to **Master-Work-Plan.md Phase 1.1**
(environment) and part of **Phase 2.1 / 3.1** (vision data + training code) —
see §2.4 below for exactly how it maps and where it diverges.

### 1.1 Taxonomy — 7 classes, decided and wired

`src/edgewaste/taxonomy.py` is the single source of truth:

```
cardboard, paper, plastic, glass, metal, organic, other
```

This **replaced** an earlier 18-class taxonomy that mixed in a medical/PPE
dataset (masks, gloves, syringes, IV bags...). That dataset is fully out of
the pipeline now — nothing references it. If you ever see a reference to
14/18 classes or `custom_dataset_dir` anywhere, it's stale/historical.

### 1.2 Vision pipeline — detect-then-classify, two independent stages

| Stage | What | Status |
|---|---|---|
| **Classifier** | ConvNeXt-Tiny + ViT-Small hybrid (attention fusion) over the 7 classes | Code complete (`src/edgewaste/{train,evaluate,infer}.py`, `models/hybrid.py`). Never trained. |
| **Detector** | YOLO26n, **single-class** (`waste_item`), localizes items before classification | Code complete (`src/edgewaste/detect/{data,train}.py`). Never trained. |
| **Combined pipeline** | detector finds+crops item → classifier types the crop | Code complete (`src/edgewaste/pipeline.py`). Never run — needs both checkpoints above. |

The classifier and detector are independent — you can train, evaluate, and
demo the classifier alone (`edgewaste-infer`) without ever touching the
detector. The detector only matters once you want conveyor-style
localization instead of single pre-framed items.

**Environment setup is already simplified vs. Master-Work-Plan.md Sub-Phase
1.1.** MWP describes a manual `venv` + long `pip install` list + `pip freeze`.
This repo already has that packaged as installable extras:

```bash
python -m venv waste-mvp-env && waste-mvp-env\Scripts\activate   # Windows
pip install -e .                 # core: torch, timm, sklearn, opencv, etc.
pip install -e ".[detect]"       # + ultralytics, for the YOLO26 detector
pip install -e ".[data]"         # + kaggle, for dataset downloads
```

`requirements.txt` still exists (MWP's `pip freeze > requirements.txt` step
is effectively already done), and `pyproject.toml` declares the same
dependencies as proper package metadata with console-script entry points —
you don't need MWP's `env_check.py` smoke test, `python -m edgewaste.taxonomy`
serves the same purpose and is already project-specific. SHAP, grad-cam, and
flwr (Flower) from MWP's Sub-Phase 1.1 list are **not yet in
`requirements.txt`** — add them when you actually reach explainability
(§3.5) and federated learning (§3.7), no need to install them now.

### 1.3 Docs reconciled

`PLAN.md` and `docs/stage-2-enhanced-system.md` in this repo have been
updated to point at the design docs and flag where the original blueprint
(`flowwork.docx`-derived docs) is now stale — see PLAN.md's "Decisions Made"
section for the full reasoning trail. This handover doc goes one step
further and reconciles against `Master-Work-Plan.md` specifically, since
that's now the most detailed execution reference available.

---

## 2. Immediate next steps — vision pipeline (Master-Work-Plan Phase 2.1 / 3.1, adapted)

### Step 0 — Kaggle credentials (blocks everything below)

1. Create a Kaggle account if you don't have one.
2. Go to kaggle.com → Settings → API → "Create New Token". This downloads `kaggle.json`.
3. Place it at `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
   Alternatively set env vars `KAGGLE_USERNAME` / `KAGGLE_KEY`.
4. Verify: `pip install kaggle` then `kaggle datasets list` should not error.

### Step 1 — Download + build the classification dataset

```bash
cd edge-waste
pip install -e .

python -m edgewaste.taxonomy        # sanity check: prints the 7 classes + 3 sources

edgewaste-fetch --config configs/stage1.yaml     # downloads all 3 Kaggle sources
edgewaste-ingest --config configs/stage1.yaml    # consolidates into data/processed/<class>/
edgewaste-split --config configs/stage1.yaml     # writes data/splits.csv (stratified 70/15/15)
```

Datasets this downloads (all Kaggle, all public, no cost):

| Source key | Kaggle slug | What it contributes |
|---|---|---|
| `trashnet` | `asdasdasasdas/garbage-classification` | cardboard, glass, metal, paper, plastic, trash→other |
| `trashbox` | `minhle13/trashbox` | same 5 material classes + e-waste/medical→other |
| `garbage12` | `mostafaabla/garbage-classification` | **organic** (its `biological` folder) — the other two sources don't have an organic class at all, so this one is not optional |

**⚠ Divergence from Master-Work-Plan.md §2.1:** MWP instructs manually
cloning TrashNet (GitHub + Google Drive mirror), TACO (GitHub + its own
`download.py` against possibly-dead Flickr URLs), TrashBox (GitHub, archive
location varies), and ZeroWaste-f (Zenodo, possibly password-protected,
tens of GB), then writing a custom `taxonomy_map.py` + `build_unified_dataset.py`
to remap and merge them. **This repo already does that job**, more simply,
via the `edgewaste-fetch`/`ingest`/`split` CLI above, using verified Kaggle
mirrors for TrashNet/TrashBox instead of MWP's fragile manual-download paths.
**Recommendation: use the CLI above, not MWP §2.1's manual clone-and-script
instructions.** The one thing MWP's dataset list has that this repo's
classification sources don't is **TACO and ZeroWaste-f as classification
sources** — this repo only uses TACO for the *detector* (see Step 3), not
the classifier, and doesn't use ZeroWaste-f at all (no confirmed Kaggle
mirror — see §4). If cross-dataset generalization reporting specifically
needs TACO/ZeroWaste-f images run through the *classifier* too, that's
extra work not yet built — flag it as a decision, don't assume it silently.

**Known risk to watch for:** `edgewaste-ingest` raises a `KeyError` naming
the exact source if a downloaded dataset's folder names don't match what's
declared in `taxonomy.py`'s `mapping` dict (Kaggle maintainers occasionally
rename folders between versions). The error message tells you which source
and which folder was unexpected — fix by adding the new folder name to that
source's `mapping` in `taxonomy.py`. Ingest already searches the whole
downloaded tree recursively (tolerates inconsistent zip nesting), so this
should only trigger on an actual rename, not just re-nesting.

### Step 2 — Train + evaluate the classifier

```bash
edgewaste-train --config configs/stage1.yaml --smoke   # 1 epoch, 64 images — sanity check first
edgewaste-train --config configs/stage1.yaml            # full run, writes runs/stage1/best.pt
edgewaste-eval --config configs/stage1.yaml --ckpt runs/stage1/best.pt
```

Target (MWP §3.1.5 / Project-Action-Plan §5): **≥80% held-out accuracy** as
a starting bar, adjust once you see real numbers. `edgewaste-eval` writes a
confusion matrix PNG and metrics JSON to `runs/stage1/`. Per MWP §3.1.4 step
7, also worth checking accuracy **per source dataset** (trashnet vs trashbox
vs garbage12) once you have real results, not just pooled — the split
manifest (`data/splits.csv`) doesn't currently carry a `source_dataset`
column through to eval, but `data/processed/provenance.csv` (written by
`edgewaste-ingest`) has the per-image source, so you can join on that if you
want this breakdown. `other` (the catch-all class) is the one most likely to
be messy in the confusion matrix since it pools unrelated categories from
all three sources.

Quick demo once trained:
```bash
edgewaste-infer --ckpt runs/stage1/best.pt path/to/some/image.jpg
edgewaste-infer --ckpt runs/stage1/best.pt --camera     # webcam, needs a camera attached
```

### Step 3 — Download + prepare the detection dataset, train the detector

```bash
pip install -e ".[detect]"                             # installs ultralytics
edgewaste-detect-fetch --config configs/detect.yaml     # downloads TACO, collapses to 1 class
edgewaste-detect-train --config configs/detect.yaml     # fine-tunes YOLO26n
```

Dataset: `vencerlanz09/taco-dataset-yolo-format` on Kaggle — a
pre-converted-to-YOLO-format mirror of TACO, avoiding MWP §2.1's
COCO-parsing + possibly-dead-Flickr-URL `download.py` path entirely. The
prep step collapses TACO's ~60 litter categories to **one class**
(`waste_item`), output checkpoint: `runs/detect/taco_single_class/weights/best.pt`.

**⚠ Architectural divergence from Master-Work-Plan.md §3.1 — flagging, not
resolving:** MWP's `waste_detect.yaml` (§3.1.4 step 2) proposes a
**7-class detector** — YOLO directly predicts `paper/cardboard/plastic/
glass/metal/organic/other`, i.e. detection and material classification
happen in one model. **This repo's actual implementation is different**:
the detector is deliberately single-class (`waste_item` only — localization
only), and material typing happens entirely in the separate classifier
stage on the cropped detection (`src/edgewaste/pipeline.py`). This was a
deliberate choice made when the detection stage was added (simpler, doesn't
need TACO/ZeroWaste-f's litter categories force-mapped onto the 7-class
recycling taxonomy, keeps detector and classifier fully swappable
independently — see `README.md`'s pipeline diagram). **If your guide
specifically wants YOLO doing the material classification itself** (closer
to MWP's version), that's a different architecture than what's built and
would mean retraining the detector as a genuine 7-class detector instead of
using `edgewaste-detect-data.py`'s current single-class collapse — a real
decision to make, not a bug to silently fix.

**Known risk:** `edgewaste-detect-fetch`'s pairing logic (matching images to
`.txt` label files by filename stem, anywhere in the downloaded tree) assumes
the Kaggle mirror is genuinely in same-stem-image+label YOLO format, which
its listing claims but hasn't been downloaded/confirmed firsthand. If
`edgewaste-detect-fetch` reports zero pairs found, inspect
`data/detect/taco_raw/` by hand — the layout may need a tweak to
`_find_pairs()` in `src/edgewaste/detect/data.py`.

### Step 4 — Run the combined pipeline

```bash
edgewaste-pipeline --det-ckpt runs/detect/taco_single_class/weights/best.pt \
    --cls-ckpt runs/stage1/best.pt path/to/image_or_dir
edgewaste-pipeline --det-ckpt ... --cls-ckpt ... --camera
```

This is the Milestone A / MWP Phase 3.1 exit criterion — trained detector +
classifier working together. Per MWP §3.1.5, target: detector mAP recorded
(no fixed target given — record and report whatever you get), classifier
≥80% test accuracy, confusion matrix errors should make physical sense
(glass↔plastic more forgivable than metal↔organic).

### Step 4.5 — Edge export + on-device benchmark (Master-Work-Plan Phase 4.1)

Once you have real checkpoints, before assuming a Pi 5 can run this in real
time:

```python
# Classifier -> ONNX (on the training machine)
import torch
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "classifier.onnx",
                   input_names=["input"], output_names=["output"], opset_version=17)
```
```python
# Detector -> ONNX (ultralytics has this built in)
from ultralytics import YOLO
model = YOLO("runs/detect/taco_single_class/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

Transfer both `.onnx` files to the Pi (`scp *.onnx <user>@<pi-ip>:~/models/`),
install `onnxruntime` in the Pi's own venv (separate from your training
machine's environment), and benchmark 50 sequential inferences on a real
captured frame (not synthetic noise) for mean/median/p95 latency. Define a
throughput target now (MWP suggests e.g. "<500ms per item end-to-end" for a
slow demo) — if it's not met, drop to detector `-n` scale (already the
default here) or apply post-training INT8 quantization via ONNX Runtime
before re-benchmarking. **Verification per MWP §4.1.5:** confirm ONNX output
matches the PyTorch model's output on the same input within `atol=1e-3` —
catches silent export bugs before they degrade on-device accuracy.

---

## 3. Everything past vision — hardware, sensors, OCI (nothing built yet)

None of what follows has any code in this repo yet. This is where
`Master-Work-Plan.md` is the primary reference — it's more detailed here
than any other doc (pin assignments, wiring safety notes, exact calibration
formulas). This section summarizes it into an executable checklist; **go to
`Master-Work-Plan.md` directly for full code snippets** when you're actually
doing each step (Sub-Phases 1.2, 1.3, 2.2, 2.3, 3.2–3.4, 4.2, 4.3).

### 3.1 Hardware bring-up (MWP Phase 1, sub-phases 1.2–1.3)

**Bill of materials (core MVP):**

| Item | Spec | Notes |
|---|---|---|
| ESP32 NodeMCU | 30-pin, CP2102 USB-UART | Sensor hub — 2 ADC channels needed |
| Capacitive moisture sensor | v1.2/v2.0 | AOUT → GPIO34 (ADC1_CH6) |
| MQ-135 gas sensor | module w/ onboard trim pot | AOUT → GPIO35 (ADC1_CH7). **Power from VIN/5V, not 3V3** — the heater needs it |
| DHT11 | temp/humidity | DATA → GPIO4, + 10kΩ pull-up to 3V3 if not onboard |
| Raspberry Pi 5 (4GB) | + official 27W USB-C PSU | A phone charger will undervoltage it |
| MicroSD card | 32GB+, Class 10/A2 | — |
| USB webcam | or Pi Camera Module 3 | — |
| Digital scale | 0.1–1g resolution | For calibration data collection (§3.3) |

**Wiring + bring-up, condensed from MWP §1.2.4/§1.3.4** (see MWP for the
full Arduino sketch and Pi Imager walkthrough):

1. Wire all three sensors per the pin table above.
2. Arduino IDE → install ESP32 board package + Adafruit DHT library.
3. Flash a bring-up sketch streaming `moisture_raw,gas_raw,temp,humidity`
   over serial (115200 baud) every 2s. Confirm no `nan` from the DHT11.
4. **MQ-135 burn-in — start this immediately, run in parallel with
   everything else in Phase 2 below:** power it continuously for 24–48h
   before any reading is meaningful. Non-negotiable per the OCI protocol
   doc Part 1.2/8.5 — pre-burn-in readings don't calibrate correctly and
   must be discarded.
5. **R0 calibration** (after burn-in, in genuinely clean/ventilated air):
   record ≥50 `gas_raw` samples over ~2 min, average, convert to voltage
   (`V_RL = gas_raw * 3.3/4095`), compute `Rs = ((5.0 - V_RL)/V_RL) * RL`
   (RL = your module's load resistor, commonly 20kΩ — check silkscreen),
   then `R0 = Rs_clean_air / 3.6`. **Hard-code this R0** — it's specific to
   your physical unit.
6. Flash Raspberry Pi OS (64-bit) via Raspberry Pi Imager, pre-configure
   SSH + Wi-Fi in the imager's settings so it's headless from first boot.
   `ssh` in, `sudo apt update && sudo apt full-upgrade -y`, enable camera
   interface if using the official module.
7. Set up a **separate** Python venv on the Pi itself (`~/waste-edge-env`)
   — this is not the same environment as your training laptop.
8. Plug the ESP32 into the Pi over USB, confirm `/dev/ttyUSB0` (or similar)
   enumerates and streams. **Verify the device path is stable across
   reboots** — if not, plan to reference it by udev rule/by-id path later.

**Deliverable:** live `{moisture_raw, gas_raw, temp, humidity}` CSV stream,
readable from the Pi over USB serial, with a documented hard-coded `R0`.

### 3.2 Public vision data — already covered in §2 above, skip MWP §2.1's manual version.

### 3.3 Physical calibration data collection (MWP Sub-Phase 2.2)

This is **real, physical data collection** — not synthetic, not skippable.
Needs the bring-up rig from §3.1, post burn-in, R0 already computed.

- **Design:** 2 categories (porous/flat: paper plate, cardboard — vs.
  rigid/enclosed: bottle, can) × 5 severity levels (0g/2g/5g/10g/20g of one
  standardized food paste/puree, kept identical across all trials) × 8
  replicates ≈ **160 samples**.
- **Log schema** (decide before collecting, not mid-collection): `trial_id,
  category, severity_level, residue_mass_g, item_reference_area_or_weight,
  moisture_raw, gas_raw, ambient_temp_c, ambient_humidity_pct, timestamp`.
- **Fixed calibration anchors** (once, not per-trial): `M_dry`/`M_wet` from
  a genuinely dry/saturated reference item; gas clean-air baseline and a
  heavily-contaminated reference reading. Save to `calibration_anchors.csv`.
- **Per trial:** weigh residue precisely, apply consistently, wait a fixed
  dwell time (e.g. 60s, same every trial) before reading, average ~10
  consecutive serial readings, log immediately.
- **Randomize severity order** within each session (not all-0g-then-all-20g)
  to avoid conflating severity with sensor drift.
- **Clean the rig between trials**, verify gas reading returns near baseline
  before the next trial.
- *Optional:* ~20–30 naturally-contaminated items for a secondary
  real-world validation set, rated 0–4 by 2–3 independent raters against a
  written rubric, inter-rater agreement via Cohen's/Fleiss' kappa.

**Verification before moving on (MWP §2.2.5):** scatter-plot `gas_raw` and
`moisture_raw` against `residue_mass_g` — both should show a visually
plausible upward trend. A flat, noise-dominated relationship means stop and
recheck wiring/dwell-time/application-consistency before going further.

### 3.4 OCI feature engineering + fitting (MWP Sub-Phases 2.3, 3.2–3.4)

```python
# Normalization — fixed-reference, not running min-max (see OCI protocol Part 1)
df["f_m"] = ((df["moisture_raw"] - M_dry) / (M_wet - M_dry)).clip(0, 1)
df["g_signal"] = -np.log(df["Rs_over_R0"])           # work in Rs/R0 space, not ppm
df["f_g"] = ((df["g_signal"] - L_min) / (L_max - L_min)).clip(0, 1)
df["severity"] = df["residue_mass_g"] / df["item_reference_area_or_weight"]
df["contaminated"] = (df["severity"] > THRESHOLD).astype(int)   # state + justify THRESHOLD
```

- **Linear vs. sigmoid for `f_m`:** fit both against real calibration data
  (`scipy.optimize.curve_fit`), keep linear unless the sigmoid's residuals
  are meaningfully lower (a real S-curve, not noise) — document whichever
  you pick with the residual comparison numbers.
- **Collinearity check first:** `df[["f_m","f_g"]].corr()`. If Pearson
  `r > 0.8`, use stronger L2 regularization (lower `C`) and/or test the
  interaction term `f_m * f_g`.
- **Fit the base model:**
  ```python
  from sklearn.linear_model import LogisticRegression
  clf = LogisticRegression(penalty="l2", C=1.0)
  clf.fit(X_train, y_train)   # X = [f_m, f_g]
  beta0, (beta1, beta2) = clf.intercept_[0], clf.coef_[0]
  ```
- **Mandatory ablation:** fit + evaluate moisture-only and gas-only models,
  compare held-out AUC against the combined model — report all three. This
  is the direct evidence that fusion beats either sensor alone.
- **Threshold selection — sensitivity-constrained, not naive Youden's**
  (a missed contamination is worse than an unnecessary reject):
  ```python
  valid_idx = np.where(tpr >= 0.95)[0]           # your minimum sensitivity target
  operating_threshold = thresholds[valid_idx[np.argmin(fpr[valid_idx])]]
  ```
  Also report the PR curve + F2-score given likely class imbalance.
- **Sensor-dropout fallback** — if one sensor fails, drop its term and
  renormalize the remaining weight rather than crashing or silently
  returning a wrong OCI; derive the renormalization scale empirically
  (compare full-sensor vs. single-sensor OCI distributions on the
  calibration set), don't guess it. Test all three availability cases
  (both, moisture-only, gas-only) on the same held-out samples — this is
  your Phase 6 sensor-dropout robustness evidence, generated early.
- **Hard-code the final `beta0..beta2(/beta3)` and `operating_threshold`**
  as constants. No live scikit-learn dependency needed at inference time —
  the deployed formula is a handful of fixed numbers.

**Deliverables:** fitted coefficients + ablation table (moisture-only /
gas-only / combined AUC) + reported Pearson `r` + ROC-AUC/PR curves + a
`compute_oci()` function handling all three sensor-availability cases
without crashing.

### 3.5 Explainability (MWP Sub-Phase 4.2)

- **Vision:** Grad-CAM on the classifier's last conv stage (`pip install
  grad-cam`). Manually inspect 10–15 outputs — highlighted regions should
  be plausible (visible residue/soiling), not background/corners.
- **OCI:** no SHAP sampling machinery needed — for a 2–3 coefficient
  logistic regression, the per-prediction contribution is just `beta1*f_m`
  and `beta2*f_g` directly. `shap.LinearExplainer` (exact, not approximate,
  for linear models) is a nice-to-have visualization layer on top, not a
  new derivation.
- Bundle both into one "explanation card" per prediction (class +
  confidence, OCI + per-sensor contribution, Grad-CAM heatmap) — reusable
  as both live-demo output and report figures.

### 3.6 End-to-end MVP integration (MWP Sub-Phase 4.3) — **this is the actual MVP milestone**

Wire camera → detector → classifier → sensors → OCI → decision into one
loop running on the Pi (`main_mvp_loop.py` in MWP §4.3.4 has a working
skeleton — reuse it, wiring in this repo's `edgewaste.pipeline` for the
vision half instead of raw ONNX calls if you'd rather stay in Python/PyTorch
than export first). No physical actuation yet — logged/displayed decision
only.

**Definition of done (MWP §4.3.5 — same numbers as `Project-Action-Plan.md`
§5):** run on 20–30 fresh items **not used in calibration**; classifier
≥80% accuracy; OCI decision meets your §3.4 sensitivity target (≥90–95%
recall on genuinely contaminated items); full cycle runs without crashing
across all items. This is the actual "system works end-to-end" checkpoint —
everything before this is a component; everything after is depth.

---

## 4. Extensions — only after §3.6's MVP is validated (MWP Phases 5–6)

Sequenced, not required. Pick based on remaining time.

| Phase | What | Reference |
|---|---|---|
| **5.1** | Category-conditional OCI weights (porous vs. rigid — refit per category); optional metal sensor (⚠ voltage mismatch — most inductive proximity sensors run 12–24V, need level-shifting before an ESP32 GPIO, don't wire directly) + load cell (HX711, 2-wire digital, no ADC channel needed) | MWP §5.1 |
| **5.2** | Physical conveyor + servo/solenoid diverter, GPIO motor control via `gpiozero`, wired into the MVP loop's decision output | MWP §5.2 |
| **5.3** | MC-Dropout uncertainty (`model.train()` at inference, N=20–30 stochastic passes) + temperature scaling + low-confidence→manual-review routing | MWP §5.3 |
| **6.1** | Flower + FedAvg federated learning, non-IID simulated bins first, then 2+ physical Pi nodes if hardware allows | MWP §6.1 |
| **6.2** | Full ablation suite (formula vs. category-conditional OCI, sensor-dropout curve, moisture/gas/combined AUC, FL vs. centralized, optional cross-attention comparison) + paper draft using the OCI protocol doc's Parts 7–8 as methodology/limitations skeleton | MWP §6.2 |

---

## 5. Things I could not verify / didn't do

- **No dataset has been downloaded** (no Kaggle credentials in this
  environment). The three classification sources' and TACO's Kaggle slugs
  are confirmed to *exist* via web search, but their exact internal folder
  structure is unverified firsthand — see the risk notes in §2 Steps 1 and 3.
- **ZeroWaste-f and WaRP-C** have **no confirmed Kaggle mirror** as of
  2026-07 — official distribution is manual download from
  `http://ai.bu.edu/zerowaste/` (Zenodo, per MWP §2.1.3, possibly
  password-protected). Not wired into any config in this repo.
- **`ultralytics` version for YOLO26** (Jan 2026 release): pinned
  `ultralytics>=8.4` as a floor, not a verified-exact version — check
  `docs.ultralytics.com/models/yolo26` if `YOLO("yolo26n.pt")` doesn't
  resolve, and bump the pin.
- **No training has been run** — every accuracy/mAP/recall number anywhere
  in this repo's docs (">80%", "≥90-95% recall") is a target, not a
  measurement.
- **No hardware has been wired, no calibration data collected, no OCI
  fitted** — all of §3 is unstarted. Nothing to verify yet.
- **The single-class-vs-7-class detector question (§2 Step 3's flagged
  divergence)** is a real open decision, not something I resolved — pick
  one deliberately before training the detector for real, since retraining
  after the fact costs GPU time and a re-labeled dataset.

---

## 6. Quick orientation for anyone new to the repo

```
edge-waste/
├── HANDOVER.md                  ← you are here
├── Master-Work-Plan.md          ← full granular execution roadmap (all phases)
├── README.md                    ← command reference
├── PLAN.md                      ← decision log + module map
├── docs/stage-{1,2,3}-*.md      ← older blueprint-derived checklists (mostly superseded)
├── configs/{stage1,detect}.yaml ← all tunables, no hardcoded paths
└── src/edgewaste/
    ├── taxonomy.py              ← the 7 classes + dataset source mappings
    ├── data/                    ← fetch → ingest → split (classification data)
    ├── detect/                  ← fetch/prepare TACO, train YOLO26 (detection data)
    ├── models/hybrid.py         ← ConvNeXt+ViT classifier architecture
    ├── train.py / evaluate.py / infer.py   ← classifier lifecycle
    └── pipeline.py              ← detector + classifier combined
```

Everything vision-related is driven by the two YAML configs — no path
should need editing in Python code itself for a normal run. Everything past
vision (§3–§4) has no code yet; `Master-Work-Plan.md` is where the wiring
diagrams, exact formulas, and full code snippets live.
