# build_review_notebooks.py - writes and runs the five review notebooks and cleans the colab training notebook (all comments in simple lowercase english)
"""usage: python scripts/report/build_review_notebooks.py
writes review/notebooks/01..06 .ipynb. notebooks 01-05 are executed here so the outputs are saved inside them."""
import json
import re
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "review" / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

SETUP = '''# find the project folder, so the notebook works from any place
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
root = Path.cwd()
while not (root / "pyproject.toml").exists():
    root = root.parent
os.chdir(root)
sys.path.insert(0, str(root / "src"))
print("project folder:", root)'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    n.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    return n


def run_and_save(name, cells):
    n = nb(cells)
    NotebookClient(n, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(n, OUT / name)
    errs = [o for c in n.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    print(name, "cells:", len(n.cells), "errors:", len(errs))


# ------------------------------------------------------------------------------------------------ notebook 1
NB1 = [
    ("md", "# 1. project overview, data and taxonomy\n\n**what this notebook shows:** the 33 classes in 9 material families, how many images each class has, how the data is split, "
           "and how the training images are changed (augmented).\n\nrun the cells from top to bottom. everything here runs in about one minute."),
    ("code", SETUP),
    ("code", '''# load the class list (the single source of truth for the project)
import pandas as pd
import matplotlib.pyplot as plt
from edgewaste import taxonomy as tx

print("number of classes :", tx.NUM_CLASSES)
print("number of families:", tx.NUM_FAMILIES)
print("families          :", list(tx.FAMILIES))
print("hazardous classes :", [c for c in tx.CLASS_NAMES if tx.is_hazardous(c)])'''),
    ("md", "## the split of the data\n\nevery image has one row in the split file: its path, its class and its group (train, val or test). the split is 70 / 15 / 15 and stays the same every time (seed 42)."),
    ("code", '''# count images per class and per group
df = pd.read_csv("data/splits_colab_reconstructed.csv")
table = df.pivot_table(index="class_name", columns="split", values="path", aggfunc="count").fillna(0).astype(int)
table["total"] = table.sum(axis=1)
table = table.loc[list(tx.CLASS_NAMES)]

print("images in each group:")
print(df["split"].value_counts().to_string())
print()
print("largest class / smallest class =", round(table["total"].max() / table["total"].min(), 1), " (this is the class imbalance)")
table.head(12)'''),
    ("code", '''# bar chart of images per class, coloured by family
colors = dict(zip(tx.FAMILIES, plt.cm.tab10.colors))
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(table)), table["total"], color=[colors[tx.CLASS_TO_FAMILY[c]] for c in table.index])
ax.set_xticks(range(len(table)))
ax.set_xticklabels([c.replace("_", " ") for c in table.index], rotation=80, ha="right", fontsize=8)
ax.set_ylabel("number of images")
ax.set_yscale("log")
ax.set_title("images per class (log scale), colour = material family")
plt.tight_layout()
plt.show()'''),
    ("code", '''# show one real image from each family
from PIL import Image
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for ax in axes.ravel():
    ax.axis("off")
for ax, fam in zip(axes.ravel(), tx.FAMILIES):
    cls = tx.FAMILY_TO_CLASSES[fam][0]
    path = df[df["class_name"] == cls]["path"].iloc[0]
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_title(f"{fam}: {cls.replace('_', ' ')}", fontsize=9)
plt.tight_layout()
plt.show()'''),
    ("md", "## why a family level?\n\na sorting machine has one bin for each *family*. two kinds of plastic bottle go to the same bin, so mixing them up does not matter for sorting. "
           "that is why the project reports **item accuracy** (which exact class) and **family accuracy** (which bin) separately, and **hazard recall** (are dangerous items found)."),
    ("md", "## how training images are changed (augmentation)\n\nthe training images are randomly cropped, flipped, rotated, colour-changed, blurred and partly erased. this copies real camera conditions. "
           "validation and test images are never changed."),
    ("code", '''# the code that builds the two image pipelines
import inspect
from edgewaste.data.waste_image_dataset import build_transforms
print(inspect.getsource(build_transforms))'''),
    ("code", '''# the same image after 6 random augmentations
import torch
train_tfm = build_transforms(224, train=True)
img = Image.open(df[df["class_name"] == "plastic_water_bottles"]["path"].iloc[5]).convert("RGB")
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
fig, axes = plt.subplots(1, 7, figsize=(16, 3))
axes[0].imshow(img.resize((224, 224)))
axes[0].set_title("original")
for ax in axes[1:]:
    x = train_tfm(img) * std + mean          # undo the normalisation so we can look at it
    ax.imshow(x.permute(1, 2, 0).clamp(0, 1))
    ax.set_title("augmented")
for ax in axes:
    ax.axis("off")
plt.show()'''),
]

# ------------------------------------------------------------------------------------------------ notebook 2
NB2 = [
    ("md", "# 2. the model code: hybrid convnext + vision transformer\n\n**what this notebook shows:** the real model code, its size, one forward pass, the training settings, "
           "and the training history of the improved recipe.\n\nidea in one line: two networks look at the same picture. convnext is good at *texture*, the vision transformer is good at *shape*. "
           "a small attention layer decides how much to trust each one for every image."),
    ("code", SETUP),
    ("code", '''# the attention layer and the full model, exactly as in the project
import inspect
from edgewaste.classification import hybrid_model as hm
print(inspect.getsource(hm.AttentionFusion))
print(inspect.getsource(hm.HybridConvNeXtViT.forward))'''),
    ("code", '''# build the model (no download needed: weights come from our own checkpoint later)
import torch
model = hm.HybridConvNeXtViT(pretrained=False)

def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6

parts = {"convnext backbone": model.convnext, "vit backbone": model.vit, "two projections": torch.nn.ModuleList([model.proj_cnx, model.proj_vit]),
         "attention fusion": model.fusion, "classifier head": model.head}
for name, part in parts.items():
    print(f"{name:20s} {n_params(part):6.2f} million parameters")
print(f"{'total':20s} {n_params(model):6.2f} million parameters")'''),
    ("code", '''# one forward pass with a random batch, to check the shapes
x = torch.randn(2, 3, 224, 224)
model.eval()
with torch.no_grad():
    logits, attn = model(x, return_attn=True)
print("logits shape   :", tuple(logits.shape), " (2 images, 33 classes)")
print("attention shape:", tuple(attn.shape), " (2 images, 2 backbones)")
print("attention adds up to one:", attn.sum(dim=1))'''),
    ("md", "## the trained model on a real image\n\nwe load the trained checkpoint and classify one real test image. the attention numbers show how much the model used each backbone."),
    ("code", '''# load the trained checkpoint
from edgewaste.config import Config
from edgewaste.classification.run_inference import load_for_inference
from edgewaste.common_utils import pick_device
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

cfg = Config.load("configs/classifier_convnext_vit.yaml")
cfg.model.pretrained = False
dev = pick_device()
model, names, tfm = load_for_inference("runs/stage1/best.pt", cfg, dev)
print("device:", dev, "| classes:", len(names))

cache = np.load("runs/analysis/convnext_vit/cache_test.npz", allow_pickle=True)
path = [p for p, l in zip(cache["paths"], cache["labels"]) if names[int(l)] == "battery"][3]
img = Image.open(path).convert("RGB")
x = tfm(img).unsqueeze(0).to(dev)
with torch.no_grad():
    logits, attn = model(x, return_attn=True)
probs = torch.softmax(logits, 1)[0].cpu().numpy()
top = probs.argsort()[::-1][:3]
plt.imshow(img); plt.axis("off")
plt.title("true class: battery")
plt.show()
for i in top:
    print(f"  {names[i]:28s} {probs[i] * 100:5.1f}%")
print("attention  convnext: %.2f   vit: %.2f" % tuple(attn[0].cpu().numpy()))'''),
    ("md", "## training settings\n\nthese are the settings of the improved recipe (`configs/classifier_convnext_vit.yaml`). key ideas:\n\n"
           "- **adamw** optimiser with a one-cycle learning rate\n- **label smoothing 0.1** so the model is not over-confident\n"
           "- **frozen backbones for 2 epochs** so the new layers learn first (this protects the pretrained features)\n"
           "- **backbones learn 10 times slower** than the new layers\n- **one** class-imbalance correction (a balanced sampler)"),
    ("code", '''# print the training part of the config file
import yaml
raw = yaml.safe_load(open("configs/classifier_convnext_vit.yaml"))
for k, v in raw["train"].items():
    print(f"{k:24s} {v}")'''),
    ("code", '''# the training function (the real code that was run on colab)
from edgewaste.classification import train_classifier as tc
print(inspect.getsource(tc.train))'''),
    ("code", '''# training history of the improved recipe (read from the saved log of the colab run)
import json
hist = json.load(open("runs/stage1_v2/history.json"))
ep = [h["epoch"] for h in hist]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ep, [h["train_loss"] for h in hist], label="train loss")
ax[0].plot(ep, [h["val_loss"] for h in hist], label="validation loss")
ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title("loss")
ax[1].plot(ep, [h["train_acc"] * 100 for h in hist], label="train accuracy")
ax[1].plot(ep, [h["val_acc"] * 100 for h in hist], label="validation accuracy")
ax[1].axvspan(0.5, 2.5, color="grey", alpha=0.2, label="backbones frozen")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("%"); ax[1].legend(); ax[1].set_title("accuracy")
plt.tight_layout(); plt.show()
best = max(hist, key=lambda h: h["val_acc"])
print(f"best validation accuracy {best['val_acc'] * 100:.2f}% at epoch {best['epoch']} of {len(hist)}")'''),
]

# ------------------------------------------------------------------------------------------------ notebook 3
NB3 = [
    ("md", "# 3. results and analysis of the trained model\n\n**what this notebook shows:** accuracy at three levels, the confusion matrix, calibration, class imbalance and the classical machine learning baselines "
           "(pca, lda, svm, knn and others). the numbers are for the first model (`runs/stage1`) on its 4,232 test images. the saved model outputs are used, so this runs in seconds."),
    ("code", SETUP),
    ("code", '''# load the saved outputs of the model on the validation and test images
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from edgewaste import taxonomy as tx
from edgewaste.analysis import calibration as cal

val = np.load("runs/analysis/convnext_vit/cache_val.npz", allow_pickle=True)
test = np.load("runs/analysis/convnext_vit/cache_test.npz", allow_pickle=True)
names = list(test["class_names"])
y_val, y_test = val["labels"].astype(int), test["labels"].astype(int)
fam_of = np.array([tx.FAMILY_TO_INDEX[tx.CLASS_TO_FAMILY[n]] for n in names])    # family number of each class
haz = tx.FAMILY_TO_INDEX["hazardous"]
print("test images:", len(y_test), "| validation images:", len(y_val))'''),
    ("md", "## accuracy at three levels\n\n- **item accuracy**: the exact class is right\n- **family accuracy**: the bin is right (this is what sorting needs)\n- **hazard recall**: of all dangerous items, how many were called dangerous"),
    ("code", '''# the three headline numbers, computed by hand so the code is easy to read
from sklearn.metrics import f1_score, balanced_accuracy_score, top_k_accuracy_score
pred = test["logits"].argmax(1)
item_acc = (pred == y_test).mean()
family_acc = (fam_of[pred] == fam_of[y_test]).mean()
is_haz = fam_of[y_test] == haz
haz_recall = (fam_of[pred][is_haz] == haz).mean()
print(f"item accuracy    : {item_acc * 100:.2f}%")
print(f"family accuracy  : {family_acc * 100:.2f}%")
print(f"hazard recall    : {haz_recall * 100:.2f}%  ({int((fam_of[pred][is_haz] == haz).sum())} of {int(is_haz.sum())})")
print(f"macro f1         : {f1_score(y_test, pred, average='macro'):.3f}")
print(f"balanced accuracy: {balanced_accuracy_score(y_test, pred) * 100:.2f}%")
print(f"top-3 accuracy   : {top_k_accuracy_score(y_test, test['logits'], k=3, labels=range(33)) * 100:.2f}%")'''),
    ("code", '''# confusion matrix at family level: rows are the true family, columns are the predicted family
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(fam_of[y_test], fam_of[pred], labels=range(len(tx.FAMILIES)), normalize="true")
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(9)); ax.set_xticklabels(tx.FAMILIES, rotation=45, ha="right")
ax.set_yticks(range(9)); ax.set_yticklabels(tx.FAMILIES)
for i in range(9):
    for j in range(9):
        ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 0.5 else "black")
ax.set_xlabel("predicted family"); ax.set_ylabel("true family"); ax.set_title("confusion matrix (% of each true family)")
plt.colorbar(im); plt.tight_layout(); plt.show()'''),
    ("code", '''# which pairs of classes get mixed up most? (almost always inside one family)
cm_item = confusion_matrix(y_test, pred, labels=range(33))
np.fill_diagonal(cm_item, 0)
pairs = np.dstack(np.unravel_index(np.argsort(-cm_item.ravel())[:8], cm_item.shape))[0]
for t, p in pairs:
    same = "same family" if fam_of[t] == fam_of[p] else "DIFFERENT family"
    print(f"{names[t]:26s} -> {names[p]:26s} {cm_item[t, p]:3d} times   ({same})")'''),
    ("md", "## calibration: can we trust the confidence?\n\nthe model is *under-confident*: it is right more often than it says. one number T (temperature), fitted on the validation set, fixes this without changing any prediction."),
    ("code", '''# fit the temperature on validation data, then check on test data
T = cal.fit_temperature(val["logits"], y_val)
before = cal.calibration_stats(cal.softmax(test["logits"]), y_test)
after = cal.calibration_stats(cal.softmax(test["logits"], T), y_test)
print(f"temperature T = {T:.2f}")
print(f"{'':22s}{'before':>10s}{'after':>10s}")
for k in ("ece", "nll", "brier", "mean_confidence", "accuracy"):
    print(f"{k:22s}{before[k]:10.3f}{after[k]:10.3f}")
print("predictions unchanged:", bool((cal.softmax(test['logits'], T).argmax(1) == pred).all()))'''),
    ("md", "## class imbalance\n\nsome classes have many more images than others. the first model corrected this twice. here we shift the scores after training with `score + tau * log(class size)` and choose tau on the validation set only."),
    ("code", '''# post-training logit adjustment
train_df = pd.read_csv("data/splits_colab_reconstructed.csv")
train_df = train_df[train_df["split"] == "train"]
n_train = train_df["class_name"].value_counts().reindex(names).values.astype(float)
def acc_with(tau, logits, y):
    return ((logits + tau * np.log(n_train)).argmax(1) == y).mean()
taus = [0, 0.5, 1.0, 1.5, 1.75, 2.0, 2.5]
rows = [(t, acc_with(t, val["logits"], y_val) * 100) for t in taus]
best_tau = max(rows, key=lambda r: r[1])[0]
print("tau   validation accuracy")
for t, a in rows:
    print(f"{t:4.2f}  {a:6.2f}%" + ("   <- chosen on validation" if t == best_tau else ""))
print(f"\\ntest accuracy: {acc_with(0, test['logits'], y_test) * 100:.2f}% before, {acc_with(best_tau, test['logits'], y_test) * 100:.2f}% after (tau = {best_tau})")'''),
    ("md", "## classical machine learning baselines (pca, lda, svm, knn, random forest ...)\n\nthe same test images were also classified with classical methods on three kinds of features. this answers: *why deep learning?* and *does the deep model beat simple methods?*"),
    ("code", '''# results of the baselines (made by src/edgewaste/analysis/classical_baselines.py)
b = pd.read_csv("reports/ml_analysis/classical_baselines.csv")
tab = b.pivot(index="model", columns="feature_set", values="test_accuracy").mul(100).round(1)
tab = tab[["handcrafted", "imagenet_frozen", "finetuned_embedding"]]
print("test accuracy (%) of each classifier on each feature set:")
display(tab)
print("our fine-tuned hybrid with its own softmax head: 89.3%")'''),
    ("code", '''# how many pca components are needed to keep 95% of the information? (fewer = more compact features)
import json
pca = json.load(open("reports/ml_analysis/summary.json"))["classical_baselines"]["pca"]
print(f"{'feature set':22s}{'raw size':>10s}{'for 90%':>10s}{'for 95%':>10s}{'for 99%':>10s}")
for name, v in pca.items():
    print(f"{name:22s}{v['raw_dim']:10d}{v['n_components_90pct']:10d}{v['n_components_95pct']:10d}{v['n_components_99pct']:10d}")
print("the fine-tuned features are very compact: 29 numbers keep 95% of the information")'''),
    ("md", "### what the baselines tell us\n\n1. hand-made features (colour + hog) with the best classical model reach only about **71%**. deep features reach **89-93%**. so deep learning is needed.\n"
           "2. naive bayes and lda have the lowest train-test gap but low accuracy (high bias). random forest and knn fit the training set almost perfectly but lose accuracy on test (high variance).\n"
           "3. an svm on **frozen** imagenet features (93.3%) was better than our first fine-tuned model (89.3%). this showed that fine-tuning had damaged the pretrained features, and led to the improved recipe (93.83% on colab)."),
]

# ------------------------------------------------------------------------------------------------ notebook 4
NB4 = [
    ("md", "# 4. safety mechanisms: uncertainty, conformal routing, cross-check, contamination index, decision engine\n\n**what this notebook shows:** the parts that make the sorter *safe*. "
           "each part is a small piece of real code you can read and run.\n\nthe main idea: the model must know when it is unsure, and a dangerous item (battery, e-waste, medical) must never go to recycling."),
    ("code", SETUP),
    ("code", '''# load the saved model outputs and the classifier head (the last small part of the network)
import numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from edgewaste import taxonomy as tx
from edgewaste.decision_engine.conformal_routing import HAZ, alpha_vector, calibrate, family_matrix, prediction_sets

val = np.load("runs/analysis/convnext_vit/cache_val.npz", allow_pickle=True)
test = np.load("runs/analysis/convnext_vit/cache_test.npz", allow_pickle=True)
names = list(test["class_names"])
y_test = test["labels"].astype(int)
M = family_matrix(names)                      # 33 x 9 table: which class belongs to which family
fam_true = M[y_test].argmax(1)

ck = torch.load("runs/stage1/best.pt", map_location="cpu", weights_only=False)
p = ck["model_cfg"]["dropout"]
head = nn.Sequential(nn.Dropout(p), nn.Linear(1024, 512), nn.GELU(), nn.Dropout(p), nn.Linear(512, 33))
head.load_state_dict({k[5:]: v for k, v in ck["model_state"].items() if k.startswith("head.")})
head.eval()
print("head loaded; dropout rate:", p)'''),
    ("md", "## 1. uncertainty with monte carlo dropout\n\nwe run the classifier head 25 times with dropout switched **on**. if the 25 answers agree, the model is sure. if they differ, it is unsure. "
           "uncertainty `U` is the normalised entropy of the average answer: 0 = sure, 1 = completely unsure."),
    ("code", '''# monte carlo dropout, written out in plain steps
@torch.no_grad()
def mc_uncertainty(emb, passes=25):
    head.train()                                               # dropout on
    probs = torch.stack([torch.softmax(head(emb), 1) for _ in range(passes)]).mean(0)
    head.eval()                                                # dropout off again
    entropy = -(probs * (probs + 1e-12).log()).sum(1)
    return probs, entropy / np.log(probs.shape[1])             # divide by log(33) so U is between 0 and 1

torch.manual_seed(0)
emb_test = torch.from_numpy(test["emb"].astype(np.float32))
_, U = mc_uncertainty(emb_test)
U = U.numpy()
pred = test["logits"].argmax(1)
wrong = pred != y_test
print(f"average uncertainty, right answers: {U[~wrong].mean():.3f}")
print(f"average uncertainty, wrong answers: {U[wrong].mean():.3f}   (higher is what we want)")

from sklearn.metrics import roc_auc_score
msp = torch.softmax(torch.from_numpy(test["logits"]), 1).max(1).values.numpy()
print(f"AUROC for finding mistakes: mc dropout {roc_auc_score(wrong, U):.3f}  |  plain max probability {roc_auc_score(wrong, -msp):.3f}")
plt.hist([U[~wrong], U[wrong]], bins=30, label=["right", "wrong"], stacked=True)
plt.xlabel("uncertainty U"); plt.legend(); plt.title("uncertainty of right and wrong predictions"); plt.show()'''),
    ("md", "## 2. conformal routing: a limit on how often a hazard reaches recycling\n\nsteps: (1) add the class probabilities into *family* totals; (2) on labelled calibration data, find for each family a threshold; "
           "(3) for a new item, keep every family whose total passes its threshold. this gives a *set* of possible families. if `hazardous` is in the set, the item is never sent to a recycling bin."),
    ("code", '''# the core of the method: thresholds per family and prediction sets (real project code)
import inspect
from edgewaste.decision_engine import conformal_routing as cr
print(inspect.getsource(cr.conformal_rank))
print(inspect.getsource(cr.calibrate))
print(inspect.getsource(cr.prediction_sets))'''),
    ("code", '''# test it: split the test images in two random halves 100 times. one half calibrates, the other half is scored.
probs = torch.softmax(torch.from_numpy(test["logits"]), 1).numpy().astype(np.float64)
fam_mass = probs @ M
top_fam = M[probs.argmax(1)].argmax(1)
is_haz = fam_true == HAZ
rng = np.random.default_rng(1)

def leakage_and_review(alpha_h, trials=100):
    leaks, reviews, leaks_gate, reviews_gate = [], [], [], []
    for _ in range(trials):
        perm = rng.permutation(len(y_test))
        cal_i, ev = perm[: len(perm) // 2], perm[len(perm) // 2:]
        q = calibrate(fam_mass[cal_i], fam_true[cal_i], alpha_vector(0.10, alpha_h))
        sets = prediction_sets(fam_mass[ev], q)
        single = sets.sum(1) == 1
        auto = single & (U[ev] < np.quantile(U[cal_i], 0.9))            # one family and low uncertainty
        auto_to_recycling = auto & (np.where(single, sets.argmax(1), -1) != HAZ)
        hz = is_haz[ev]
        leaks.append((hz & auto_to_recycling).sum() / hz.sum())
        reviews.append((~auto).mean())
        # baseline: plain max-probability gate with the same review share
        thr = np.quantile(msp[cal_i], np.mean(~auto))
        ok = msp[ev] >= thr
        leaks_gate.append((hz & ok & (top_fam[ev] != HAZ)).sum() / hz.sum())
    return np.mean(leaks) * 100, np.mean(reviews) * 100, np.mean(leaks_gate) * 100

top1_leak = (is_haz & (top_fam != HAZ)).sum() / is_haz.sum() * 100
print(f"top-1 routing with no gate: {top1_leak:.2f}% of hazards reach a recycling bin\\n")
print(f"{'alpha_H':>8s} {'conformal leak %':>17s} {'items to review %':>18s} {'max-prob gate leak %':>21s}")
for a in (0.05, 0.02, 0.01):
    l, r, lg = leakage_and_review(a)
    print(f"{a:8.2f} {l:17.2f} {r:18.1f} {lg:21.2f}")'''),
    ("md", "read the table: the operator picks `alpha_H` (the allowed hazard leakage). the measured leakage stays below it, and a smaller `alpha_H` costs more items sent to a person. "
           "the guarantee is for items that look like the calibration items. for very different images the uncertainty gate gives extra protection."),
    ("md", "## 3. object-material cross-check that never weakens a hazard\n\nthe object detector says *what the object is* (for example a can). a can is not made of plastic, so plastic gets a lower weight. "
           "classes in the hazardous family always keep full weight."),
    ("code", '''# a real example: an aluminium soda can that the classifier first calls a plastic bottle
import inspect
from edgewaste.decision_engine import object_identity_prior as oip
print(inspect.getsource(oip.apply_identity_prior))
can = names.index("aluminum_soda_cans")
idx = [i for i in range(len(y_test)) if y_test[i] == can and pred[i] != can][0]
p0 = torch.from_numpy(probs[idx]).float()
p1 = oip.apply_identity_prior(p0, "Can")
for label, pr in (("before cross-check", p0), ("after cross-check ", p1)):
    top = pr.argsort(descending=True)[:2]
    print(label, "->", ", ".join(f"{names[i]} {pr[i] * 100:.0f}%" for i in top))
print("is 'battery' a consistent reading for a Can?", oip.is_consistent("Can", "battery"), " (hazards are never treated as a contradiction)")'''),
    ("md", "## 4. organic contamination index (oci) from two sensors\n\nmoisture and gas readings are scaled to 0-1 with fixed anchors, then three small models are fitted: both sensors, moisture only, gas only. "
           "if a sensor fails, the matching model is used. the threshold is chosen so that at least 95% of contaminated items are found. **the sensor data here is simulated** because the physical sensors are not built yet."),
    ("code", '''# fit the three models on simulated calibration data
from edgewaste.contamination.synthetic_calibration_data import generate_synthetic_calibration_data
from edgewaste.contamination.sensor_normalization import normalize_gas, normalize_moisture
from edgewaste.contamination.oci_model import fit_oci_weights, compute_oci, select_threshold

train_df, anchors = generate_synthetic_calibration_data(n_replicates=300, seed=0)
test_df, _ = generate_synthetic_calibration_data(n_replicates=300, seed=1, anchors=anchors)

def features(df):
    fm = np.array([normalize_moisture(m, anchors) for m in df.moisture_raw])
    fg = np.array([normalize_gas(r, anchors) if False else float(np.clip((-np.log(r) - anchors.l_min) / (anchors.l_max - anchors.l_min), 0, 1)) for r in df.rs_over_r0])
    return fm, fg

fm_tr, fg_tr = features(train_df); y_tr = train_df.contaminated.values.astype(int)
fm_te, fg_te = features(test_df);  y_te = test_df.contaminated.values.astype(int)
model, info = fit_oci_weights(fm_tr, fg_tr, y_tr, use_interaction=True)
print("model coefficients:", {k: round(v, 3) for k, v in model.__dict__.items() if k != "operating_threshold"})

def score(fm, fg, use_m=True, use_g=True):
    return np.array([compute_oci(model, a if use_m else None, b if use_g else None) for a, b in zip(fm, fg)])

cases = {"both sensors": (True, True), "moisture only": (True, False), "gas only": (False, True)}
print(f"\\n{'case':15s} {'threshold':>9s} {'sensitivity':>12s} {'false positive rate':>20s}")
for name, (um, ug) in cases.items():
    thr = select_threshold(y_tr, score(fm_tr, fg_tr, um, ug), 0.95)["operating_threshold"]
    s = score(fm_te, fg_te, um, ug)
    print(f"{name:15s} {thr:9.3f} {(s[y_te == 1] >= thr).mean() * 100:11.1f}% {(s[y_te == 0] >= thr).mean() * 100:19.1f}%")'''),
    ("code", '''# what if the gas sensor breaks? compare the dedicated model with "put zero for the missing gas value"
thr_both = select_threshold(y_tr, score(fm_tr, fg_tr), 0.95)["operating_threshold"]
thr_m = select_threshold(y_tr, score(fm_tr, fg_tr, True, False), 0.95)["operating_threshold"]
dedicated = score(fm_te, fg_te, True, False)
default_zero = score(fm_te, np.zeros_like(fg_te))            # wrong way: pretend gas = 0
print(f"gas lost, dedicated moisture model + own threshold : sensitivity {(dedicated[y_te == 1] >= thr_m).mean() * 100:.1f}%")
print(f"gas lost, gas = 0 in the big model + its threshold : sensitivity {(default_zero[y_te == 1] >= thr_both).mean() * 100:.1f}%   (target was 95%)")'''),
    ("md", "## 5. the decision engine\n\nit combines the item, the uncertainty, the family set and the contamination score and returns one action."),
    ("code", '''# four example items through the real decision function
from edgewaste.decision_engine.routing_rules import decide
cases = [
    ("plastic_water_bottles", 0.92, 0.17, 0.10, {"plastic"}),
    ("battery", 0.84, 0.30, 0.10, {"hazardous"}),
    ("battery", 0.45, 0.80, 0.10, {"hazardous", "metal"}),
    ("cardboard_boxes", 0.88, 0.20, 0.90, {"cardboard"}),
    ("aluminum_soda_cans", 0.20, 0.81, 0.10, set()),
]
for name, conf, u, oci, fam in cases:
    d = decide(name, conf, u, oci, uncertainty_threshold=0.73, oci_threshold=0.5, family_set=fam)
    print(f"{name:24s} U={u:.2f} OCI={oci:.2f} set={sorted(fam)!s:24s} -> {d.route}")'''),
]

# ------------------------------------------------------------------------------------------------ notebook 5
NB5 = [
    ("md", "# 5. live demo: full pipeline on real images, explanations, federated learning and video\n\n**what this notebook shows:** the working system. it loads the trained model, classifies real test images, "
           "makes decisions, explains one decision with a heat map, runs a small federated learning simulation and shows the video survey result."),
    ("code", SETUP),
    ("code", '''# load the trained model and calibrate the thresholds on the validation set (never on the test set)
import numpy as np, torch
import matplotlib.pyplot as plt
from PIL import Image
from edgewaste import taxonomy as tx
from edgewaste.config import Config
from edgewaste.common_utils import pick_device
from edgewaste.classification.run_inference import load_for_inference
from edgewaste.decision_engine.conformal_routing import alpha_vector, calibrate, family_matrix, prediction_sets
from edgewaste.decision_engine.routing_rules import decide

cfg = Config.load("configs/classifier_convnext_vit.yaml")
cfg.model.pretrained = False
dev = pick_device()
model, names, tfm = load_for_inference("runs/stage1/best.pt", cfg, dev)
head = model.head
M = family_matrix(list(names))

val = np.load("runs/analysis/convnext_vit/cache_val.npz", allow_pickle=True)
test = np.load("runs/analysis/convnext_vit/cache_test.npz", allow_pickle=True)
probs_val = torch.softmax(torch.from_numpy(val["logits"].astype(np.float32)), 1).numpy().astype(np.float64)
q = calibrate(probs_val @ M, M[val["labels"].astype(int)].argmax(1), alpha_vector(0.10, 0.05))

@torch.no_grad()
def mc(emb, passes=25):
    head.train()
    p = torch.stack([torch.softmax(head(emb), 1) for _ in range(passes)]).mean(0)
    head.eval()
    return -(p * (p + 1e-12).log()).sum(1) / np.log(p.shape[1])

tau_u = float(np.quantile(mc(torch.from_numpy(val["emb"].astype(np.float32)).to(dev)).cpu().numpy(), 0.90))
print(f"device {dev} | uncertainty threshold {tau_u:.3f} | hazard family threshold {q[tx.FAMILY_TO_INDEX['hazardous']]:.3f}")'''),
    ("code", '''# the whole pipeline for one image: classify, measure uncertainty, build the family set, decide
@torch.no_grad()
def run_pipeline(img):
    x = tfm(img.convert("RGB")).unsqueeze(0).to(dev)
    model.eval()
    emb = model.feature_vector(x)
    p = torch.softmax(head(emb), 1)[0].cpu().numpy().astype(np.float64)
    u = float(mc(emb)[0])
    fam_set = {tx.FAMILIES[i] for i in np.where(prediction_sets((p @ M)[None, :], q)[0])[0]}
    top = int(p.argmax())
    return names[top], float(p[top]), u, fam_set, decide(names[top], p[top], u, None, uncertainty_threshold=tau_u, family_set=fam_set)

# pick 8 test images: a mix of normal items and hazards
rng = np.random.default_rng(3)
wanted = ["plastic_water_bottles", "newspaper", "glass_food_jars", "aluminum_soda_cans", "food_waste", "battery", "e_waste", "medical"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
for ax, cls in zip(axes.ravel(), wanted):
    idx = np.where(test["labels"] == names.index(cls))[0]
    path = test["paths"][int(rng.choice(idx))]
    img = Image.open(path).convert("RGB")
    name, conf, u, fam_set, d = run_pipeline(img)
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"true: {cls.replace('_', ' ')}\\nmodel: {name.replace('_', ' ')} ({conf * 100:.0f}%)\\nU={u:.2f} set={sorted(fam_set)}\\nDECISION: {d.route.upper()}", fontsize=9)
plt.tight_layout(); plt.show()'''),
    ("md", "## explain a decision with grad-cam\n\nthe heat map shows which parts of the picture pushed the model towards its answer (red = strong)."),
    ("code", '''# grad-cam on one image (explains the convnext part of the model)
from edgewaste.explainability.gradcam_explanation import build_gradcam, gradcam_overlay
cam = build_gradcam(model)
path = test["paths"][int(rng.choice(np.where(test["labels"] == names.index("battery"))[0]))]
img = Image.open(path).convert("RGB")
rgb, overlay = gradcam_overlay(cam, model, tfm, img, dev, class_idx=names.index("battery"))
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
ax[0].imshow(rgb); ax[0].set_title("input"); ax[1].imshow(overlay); ax[1].set_title("grad-cam heat map")
for a in ax:
    a.axis("off")
plt.show()'''),
    ("md", "## federated learning (small simulation)\n\nfive sorting units each keep their own images and different class mixes. in every round they train a local copy and send **only parameters** to a server, which averages them: "
           "`new model = sum( (images of unit / all images) * parameters of unit )`. for speed this uses the saved image features and trains only the classifier layers."),
    ("code", '''# federated averaging on saved features: 5 units, uneven data, 8 rounds
import torch.nn as nn
train = np.load("runs/analysis/convnext_vit/cache_train.npz", allow_pickle=True)
Xtr = torch.from_numpy(train["emb"].astype(np.float32)); ytr = torch.from_numpy(train["labels"].astype(np.int64))
Xte = torch.from_numpy(test["emb"].astype(np.float32)); yte = torch.from_numpy(test["labels"].astype(np.int64))
K, ROUNDS, LOCAL_EPOCHS = 5, 8, 2
torch.manual_seed(0); rng = np.random.default_rng(0)

# give every unit a different mix of classes (dirichlet split)
parts = [[] for _ in range(K)]
for c in range(33):
    ids = np.where(ytr.numpy() == c)[0]
    share = rng.dirichlet([0.3] * K)
    cuts = (np.cumsum(share) * len(ids)).astype(int)[:-1]
    for k, chunk in enumerate(np.split(rng.permutation(ids), cuts)):
        parts[k] += chunk.tolist()
sizes = [len(p) for p in parts]
print("images per unit:", sizes)

def new_model():
    return nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 33))

def local_train(m, ids, epochs):
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    ids = torch.tensor(ids)
    for _ in range(epochs):
        order = ids[torch.randperm(len(ids))]
        for b in range(0, len(order), 128):
            i = order[b:b + 128]
            opt.zero_grad(); nn.functional.cross_entropy(m(Xtr[i]), ytr[i]).backward(); opt.step()

def accuracy(m):
    return float((m(Xte).argmax(1) == yte).float().mean())

global_model, curve = new_model(), []
for r in range(ROUNDS):
    states = []
    for k in range(K):
        local = new_model(); local.load_state_dict(global_model.state_dict())
        local_train(local, parts[k], LOCAL_EPOCHS)
        states.append(local.state_dict())
    # weighted average of the parameters of all units
    avg = {key: sum(states[k][key] * (sizes[k] / sum(sizes)) for k in range(K)) for key in states[0]}
    global_model.load_state_dict(avg)
    curve.append(accuracy(global_model))
    print(f"round {r + 1}: shared model accuracy {curve[-1] * 100:.1f}%")

central = new_model(); local_train(central, list(range(len(ytr))), 8)
alone = []
for k in range(K):
    m = new_model(); local_train(m, parts[k], LOCAL_EPOCHS * ROUNDS); alone.append(accuracy(m))
print(f"\\none central model: {accuracy(central) * 100:.1f}%  |  units training alone: {np.mean(alone) * 100:.1f}% on average")'''),
    ("md", "## video litter survey\n\na tracker gives every object an id, so one object seen in many frames is counted once. below is the saved result for a 240-frame test video made from 12 real litter photographs."),
    ("code", '''# the inventory made by the video pipeline
import pandas as pd
print(open("reports/demo/video/inventory.txt").read()[:1400])
tracks = pd.read_csv("reports/demo/video/tracks.csv")
print("inventory entries:", len(tracks), "| per-frame detections merged:", int(tracks["frames_visible"].sum()))
tracks[["track_id", "object_class", "material", "uncertainty", "route", "frames_visible"]].head(8)'''),
]


# ------------------------------------------------------------------------------------------------ notebook 6 (colab)
def clean_colab():
    src = ROOT / "notebooks" / "colab_classifier_training.ipynb"
    n = json.loads(src.read_text(encoding="utf-8"))
    for c in n["cells"]:
        if c["cell_type"] != "code":
            continue
        out = []
        for line in "".join(c["source"]).split("\n"):
            m = re.match(r"^(\s*)#(.*)$", line)
            out.append(f"{m.group(1)}#{m.group(2).lower()}" if m else line)
        c["source"] = "\n".join(out)
    first = n["cells"][1]
    first["source"] = "".join(first["source"]) if isinstance(first["source"], list) else first["source"]
    note = ("\n\n**recorded result of the real run (colab t4, 12 epochs, 48 minutes):** item accuracy **93.83%**, family accuracy **97.28%**, hazard recall **98.37%** "
            "(best validation accuracy 93.86% at epoch 8). the first recipe gave 89.30% / 94.14% / 93.50%. see notebook 02 for the training curves.")
    if "recorded result" not in first["source"]:
        first["source"] += note
    (OUT / "06_colab_training.ipynb").write_text(json.dumps(n, indent=1, ensure_ascii=False), encoding="utf-8")
    print("06_colab_training.ipynb written")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    jobs = {"1": ("01_data_and_taxonomy.ipynb", NB1), "2": ("02_model_code_and_training.ipynb", NB2), "3": ("03_results_and_analysis.ipynb", NB3),
            "4": ("04_safety_mechanisms.ipynb", NB4), "5": ("05_live_demo_federated_video.ipynb", NB5)}
    for w in which:
        if w == "6":
            clean_colab()
        else:
            run_and_save(*jobs[w])
