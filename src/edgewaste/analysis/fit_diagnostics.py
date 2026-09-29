# fit_diagnostics.py - overfitting / underfitting evidence: generalisation gap, learning curve, validation curve, epoch history
"""Is the model overfitting or underfitting? Four independent pieces of evidence:

1. Generalisation gap of the deep model — accuracy and loss on train (scored
   in eval mode, no augmentation) vs validation vs test. A large train-test gap
   with near-perfect train accuracy = variance (overfitting); low train
   accuracy = bias (underfitting). val ~= test also shows model selection on
   val did not overfit the validation set.
2. Learning curve — a classifier on frozen features trained on growing
   fractions of the training set: if validation accuracy is still rising at
   100% of the data, more data would help (variance-limited); if train and
   validation have converged to a low value, capacity is the limit (bias).
3. Validation curve — the same classifier across regularisation strength C,
   showing the underfitting (low C) -> good fit -> overfitting (high C) regimes.
4. Training history — per-epoch train/val loss and accuracy, when a
   history.json from the training run is available, plus the early-stopping
   epoch recorded in the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit


def generalisation_gap(splits: dict[str, dict]) -> dict:
    out = {}
    for name, d in splits.items():
        y, p = d["labels"].astype(int), d["probs"]
        out[name] = {"accuracy": float((p.argmax(1) == y).mean()),
                     "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
                     "n": int(len(y))}
    out["train_minus_test_accuracy"] = out["train"]["accuracy"] - out["test"]["accuracy"]
    out["val_minus_test_accuracy"] = out["val"]["accuracy"] - out["test"]["accuracy"]
    return out


def per_class_gap(train: dict, test: dict, class_names: list[str]) -> list[dict]:
    rows = []
    for i, n in enumerate(class_names):
        tr = train["labels"] == i
        te = test["labels"] == i
        a_tr = float((train["probs"][tr].argmax(1) == i).mean())
        a_te = float((test["probs"][te].argmax(1) == i).mean())
        rows.append({"class": n, "train_recall": a_tr, "test_recall": a_te, "gap": a_tr - a_te,
                     "train_support": int(tr.sum())})
    return sorted(rows, key=lambda r: -r["gap"])


def learning_curve(Xtr, ytr, Xva, yva, fractions=(0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0), C=0.1, seed=0):
    rows = []
    for f in fractions:
        if f < 1.0:
            idx, _ = next(StratifiedShuffleSplit(1, train_size=f, random_state=seed).split(Xtr, ytr))
        else:
            idx = np.arange(len(ytr))
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr[idx], ytr[idx])
        rows.append({"fraction": f, "n_train": int(len(idx)),
                     "train_accuracy": float((clf.predict(Xtr[idx]) == ytr[idx]).mean()),
                     "val_accuracy": float((clf.predict(Xva) == yva).mean())})
    return rows


def validation_curve(Xtr, ytr, Xva, yva, Cs=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    rows = []
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)
        rows.append({"C": C, "train_accuracy": float((clf.predict(Xtr) == ytr).mean()),
                     "val_accuracy": float((clf.predict(Xva) == yva).mean())})
    return rows


def checkpoint_metadata(ckpt_path: str) -> dict:
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {"best_epoch": ck.get("epoch"), "best_val_accuracy_logged": ck.get("metric")}


def load_history(path: str | Path) -> list[dict] | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def plot_fit(gap: dict, lc: list[dict], vc: list[dict], history: list[dict] | None,
             history_label: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4 if history else 3, figsize=(19 if history else 15, 4.3))
    names = ["train", "val", "test"]
    axes[0].bar(names, [gap[n]["accuracy"] for n in names], color=["#4c78a8", "#f58518", "#54a24b"])
    for i, n in enumerate(names):
        axes[0].text(i, gap[n]["accuracy"] + 0.005, f"{gap[n]['accuracy'] * 100:.2f}%", ha="center")
    axes[0].set_ylim(0.8, 1.02); axes[0].set_title("Deep model accuracy per split\n(eval mode, no augmentation)")

    axes[1].plot([r["n_train"] for r in lc], [r["train_accuracy"] for r in lc], "o-", label="train")
    axes[1].plot([r["n_train"] for r in lc], [r["val_accuracy"] for r in lc], "o-", label="validation")
    axes[1].set_xscale("log"); axes[1].set_xlabel("training images"); axes[1].legend()
    axes[1].set_title("Learning curve\n(logistic regression, frozen ImageNet features)")

    axes[2].plot([r["C"] for r in vc], [r["train_accuracy"] for r in vc], "o-", label="train")
    axes[2].plot([r["C"] for r in vc], [r["val_accuracy"] for r in vc], "o-", label="validation")
    axes[2].set_xscale("log"); axes[2].set_xlabel("C (inverse regularisation strength)"); axes[2].legend()
    axes[2].set_title("Validation curve\n(underfit <- C -> overfit)")

    if history:
        ep = [h["epoch"] for h in history]
        ax = axes[3]
        ax.plot(ep, [h["train_loss"] for h in history], "o-", label="train loss")
        ax.plot(ep, [h["val_loss"] for h in history], "o-", label="val loss")
        ax2 = ax.twinx()
        ax2.plot(ep, [h["val_acc"] for h in history], "s--", color="#54a24b", label="val acc")
        ax.set_xlabel("epoch"); ax.legend(loc="upper right", fontsize=8); ax2.legend(loc="center right", fontsize=8)
        ax.set_title(f"Training history\n({history_label})")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=125); plt.close(fig)
