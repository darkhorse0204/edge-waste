# annotate_colab_notebook.py - adds simple lowercase comments and "where is the code / why" notes to the executed colab notebook, without changing code or outputs
"""usage: python scripts/report/annotate_colab_notebook.py
reads  notebooks/colab_classifier_training_executed.ipynb   (the real run, left untouched)
writes review/notebooks/06_colab_training_executed_annotated.ipynb"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "notebooks" / "colab_classifier_training_executed.ipynb"
DST = ROOT / "review" / "notebooks" / "06_colab_training_executed_annotated.ipynb"

# cell number -> list of (text that starts a line in that cell, comment to put above it)
COMMENTS = {
    3: [("from google.colab import drive", "# step 0: connect google drive, so data and checkpoints survive if colab disconnects"),
        ("import torch, shutil", "# check that a gpu is present (training on a cpu would take days)"),
        ("total, used, free", "# check free drive space (datasets need about 4 gb, each checkpoint about 200 mb)")],
    5: [("REPO_SLUG", "# the project code is in a private github repo; this cell downloads the latest version of it"),
        ("TOKEN, _why", "# read the github token from colab secrets (it is never typed in the notebook)"),
        ("AUTH_URL", "# the token is used only inside this url for git, and is never saved on drive"),
        ("def sh(", "# small helper: run a shell command and hide the token in anything it prints"),
        ("def sync_in_place", "# make the drive copy exactly equal to the latest github code (hard reset)"),
        ("has_dir = ", "# three cases: checkout exists / folder exists without git / nothing exists yet"),
        ("cfg_path = os.path.join", "# safety check: the token must not be stored in the git config on drive")],
    6: [("shutil.rmtree", "# delete old processed images and the old split, so stale duplicate images cannot enter the new split (this fixed the 1,234-steps bug)")],
    8: [("!pip install", "# install our package (edgewaste) in editable mode, so colab runs the code from the repo")],
    10: [("cfg = cf.Config.load", "# load the training settings from the yaml file"),
         ("assert tx.NUM_CLASSES", "# gate 1: stop if the code is old (it must have 33 classes)"),
         ("assert cfg.model.pretrained", "# gate 2: stop if pretrained weights are off (training from scratch gave only 49%)")],
    14: [("src_json =", "# kaggle needs a key file (kaggle.json) to allow the dataset download")],
    16: [("!edgewaste-fetch", "# download the 3 public datasets (code: src/edgewaste/data/download_kaggle_datasets.py)")],
    18: [("!edgewaste-ingest", "# ingest: copy images into one folder per class using the mapping in taxonomy.py; unreadable images are skipped"),
         ("!edgewaste-split", "# split: stratified 70/15/15 train/val/test with seed 42 (code: src/edgewaste/data/make_data_splits.py)")],
    19: [("df = pd.read_csv", "# read the split file and count images per class"),
         ("n_prov", "# gate 3: the split must contain exactly the images of the latest ingest (no stale duplicates)"),
         ("empty =", "# gate 4: every one of the 33 classes must have images")],
    21: [("!edgewaste-train", "# smoke test: 1 epoch on a tiny subset, only to check that everything is connected")],
    23: [("!edgewaste-train", "# the real training: runs train() in src/edgewaste/classification/train_classifier.py with the improved recipe from the yaml file")],
    25: [("!edgewaste-eval", "# score the best checkpoint on the test split: item level, family level and hazard recall (code: src/edgewaste/classification/evaluate_classifier.py)")],
    26: [("for p in", "# show the two confusion matrices saved by the evaluation")],
    28: [("m_path", "# read the saved metrics file and print the three headline numbers"),
         ("print('\\n=== FILES", "# list the files to download from drive")],
}

# cell number -> markdown note inserted after that cell: what happens, where the code is, why it is done
NOTES = {
    3: "**what happens:** drive is mounted and the gpu is checked (a Tesla T4 in this run, 66.5 GB free).\n\n**why:** training takes about an hour, so checkpoints must be saved on drive, and a gpu is needed.",
    5: "**what happens:** the repo is brought to the newest version of the code.\n\n**why a hard reset:** a normal `git pull` can fail on a conflict and silently leave old code, which would train the old 7-class taxonomy without any error.",
    6: "**what happens:** old processed images and the old split file are deleted; only the downloads stay.\n\n**why:** an earlier run showed 1,234 steps per epoch instead of 617, because old copies of every image were still on drive and the same photo could land in both train and test. deleting them makes the split clean again.",
    8: "**what happens:** `pip install -e .` makes the folder `src/edgewaste/` importable and creates the commands `edgewaste-fetch`, `edgewaste-ingest`, `edgewaste-split`, `edgewaste-train`, `edgewaste-eval` (they are listed in `pyproject.toml`).",
    10: "**what happens:** the class list and the training config are loaded and two safety gates run.\n\n**where:** classes are defined in `src/edgewaste/taxonomy.py`; settings are read by `Config.load` in `src/edgewaste/config.py` from `configs/classifier_convnext_vit.yaml`.\n\n**why the gates:** a wrong class list or `pretrained: false` would not give an error, only a bad model.",
    16: "**what happens:** three public kaggle datasets are present (`[have]` means they were already downloaded).\n\n**where:** `src/edgewaste/data/download_kaggle_datasets.py`; dataset names are in `src/edgewaste/taxonomy.py` (`SOURCES`).",
    18: "**what happens:** images from the 3 sources are mapped to the 33 classes, one unreadable image is skipped, then the 70/15/15 split is written to `data/splits.csv`.\n\n**where:** `ingest_source` in `src/edgewaste/data/ingest_raw_datasets.py` and `main` in `src/edgewaste/data/make_data_splits.py`.\n\n**why stratified:** every class keeps the same share in train, validation and test, so rare classes are tested too.",
    19: "**what happens:** the split is counted per family and class, and three checks run (no stale duplicates, all 33 classes present, no class with fewer than 50 images).\n\n**why:** these problems do not raise an error by themselves; they only make the model quietly worse.",
    21: "**what happens:** one tiny epoch. the values (accuracy 1.0 on a few images) mean nothing; the aim is only to prove that data, model, loss, optimiser and saving all work. note `Trainable parameters: 50,752,002` (the model size).",
    23: "**what happens:** the real training, with these ideas from `configs/classifier_convnext_vit.yaml`:\n\n"
        "| idea | where in `src/edgewaste/classification/train_classifier.py` |\n|---|---|\n"
        "| balanced batches (one imbalance correction) | `WeightedRandomSampler`, line 53 |\n"
        "| cross-entropy with label smoothing 0.1 | `nn.CrossEntropyLoss(... label_smoothing ...)`, line 105 |\n"
        "| backbones learn 10x slower than new layers | two parameter groups, lines 108-110 |\n"
        "| AdamW optimiser | line 111 |\n| one-cycle learning rate (warm-up then slow decay) | `OneCycleLR`, line 116 |\n"
        "| mixed precision (faster on the gpu) | `GradScaler` and `autocast`, lines 122 and 143 |\n"
        "| backbones frozen for the first 2 epochs | `frozen = epoch <= freeze_backbone_epochs`, line 131 |\n"
        "| gradient clipping at 1.0 | `clip_grad_norm_`, line 149 |\n| keep the best model, early stopping | best validation accuracy is saved to `best.pt`; stop after 5 epochs without improvement |\n\n"
        "**what the log shows:** validation accuracy is already 0.900 after epoch 1 (only the new layers trained). it is best at epoch 8 (0.939), and training stopped after epoch 13 because there was no improvement for 5 epochs. training accuracy keeps rising (0.975) while validation stays near 0.938, a small gap, which is mild overfitting but not harmful.",
    25: "**what happens:** the best checkpoint is scored on the 4,232 test images.\n\n**where:** `evaluate` in `src/edgewaste/classification/evaluate_classifier.py`; family level in `_family_level` (line 141); hazard recall in `_hazard_recall` (line 166).\n\n"
        "**how to read the result:** item accuracy 0.9383 (exact class), family accuracy 0.9728 (correct bin), hazard recall 0.9837 (726 of 738 dangerous items were called dangerous). the 12 missed hazards are listed with where they went (mostly office paper).\n\n"
        "**note:** the warnings about `DataLoader` workers and `Palette images` are harmless; they do not change results.",
    28: "**what happens:** the saved metrics are printed again as the final summary: **93.83% item, 97.28% family, 98.37% hazard recall** on 4,232 test images, measured on colab's own split.",
}

HEAD = """# Annotated executed notebook: training the 33-class classifier on Colab

This is the **real executed run** (outputs are kept exactly as Colab produced them). Only comments and short notes were added; no code line was changed.
The original file is `notebooks/colab_classifier_training_executed.ipynb`.

**result of this run:** 13 epochs (early stopping), about 52 minutes on a Tesla T4. best validation accuracy 93.86% at epoch 8. test set (4,232 images): item accuracy **93.83%**, family accuracy **97.28%**, hazard recall **98.37%** (726 of 738).

## where is each feature coded?

| feature | file (inside `src/edgewaste/`) | main function or class |
|---|---|---|
| 33 classes, 9 families, data sources | `taxonomy.py` | `CLASSES`, `FAMILIES`, `SOURCES` |
| settings from yaml | `config.py` | `Config.load`, `TrainConfig` |
| download, ingest, split | `data/download_kaggle_datasets.py`, `data/ingest_raw_datasets.py`, `data/make_data_splits.py` | `ingest_source`, `main` |
| image loading, augmentation, sampler weights | `data/waste_image_dataset.py` | `build_transforms`, `WasteDataset.sampler_weights` |
| hybrid model (convnext + vit + attention) | `classification/hybrid_model.py` | `AttentionFusion`, `HybridConvNeXtViT` |
| training loop | `classification/train_classifier.py` | `train` |
| evaluation (item, family, hazard) | `classification/evaluate_classifier.py` | `evaluate`, `_family_level`, `_hazard_recall` |
| uncertainty (monte carlo dropout) | `decision_engine/mc_dropout_uncertainty.py` | `mc_dropout_predict` |
| conformal family-set routing | `decision_engine/conformal_routing.py` | `calibrate`, `prediction_sets` |
| object-material cross-check | `decision_engine/object_identity_prior.py` | `apply_identity_prior` |
| final decision | `decision_engine/routing_rules.py` | `decide` |
| contamination index (sensors) | `contamination/oci_model.py`, `contamination/sensor_normalization.py` | `fit_oci_weights`, `compute_oci` |

Notes with **what / where / why** are added under the important cells. Comments inside the code cells are in simple lowercase english.
"""


def lower_comment(line):
    m = re.match(r"^(\s*)#(.*)$", line)
    if m:
        return f"{m.group(1)}#{m.group(2).lower()}"
    m = re.match(r"^(.*\S)(\s{2,})#(.*)$", line)
    if m and not re.search(r"['\"]", m.group(3)) and m.group(1).count("'") % 2 == 0 and m.group(1).count('"') % 2 == 0:
        return f"{m.group(1)}{m.group(2)}#{m.group(3).lower()}"
    return line


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    new_cells = [md(HEAD)]
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            lines = "".join(c["source"]).split("\n")
            out = []
            for ln in lines:
                for anchor, comment in COMMENTS.get(i, []):
                    if ln.lstrip().startswith(anchor):
                        indent = ln[: len(ln) - len(ln.lstrip())]
                        out.append(f"{indent}{comment}")
                out.append(lower_comment(ln))
            c["source"] = "\n".join(out)
        new_cells.append(c)
        if i in NOTES:
            new_cells.append(md(NOTES[i]))
    nb["cells"] = new_cells
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("written", DST, "cells:", len(new_cells))


if __name__ == "__main__":
    main()
