"""Evaluate a trained checkpoint on the test split (Stage 1 metrics).

Reports accuracy, macro/weighted precision/recall/F1, a per-class table, and
writes a confusion matrix PNG. These are the section-14 metrics relevant to a
camera-only classifier; inference-time/FPS/energy come in Stage 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import WasteDataset
from .models import build_model
from .utils import pick_device


def _load_model(ckpt_path: Path, cfg: Config, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg.model, num_classes=len(ckpt["class_names"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["class_names"]


@torch.no_grad()
def evaluate(cfg: Config, ckpt: str, split: str = "test") -> dict:
    device = pick_device()
    ckpt_path = Path(ckpt)
    model, class_names = _load_model(ckpt_path, cfg, device)

    ds = WasteDataset(cfg.data.manifest, split, cfg.model.image_size, train=False)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers)

    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())

    y_true_a, y_pred_a = np.array(y_true), np.array(y_pred)
    acc = float((y_true_a == y_pred_a).mean())
    present = sorted(set(y_true))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_a, y_pred_a, labels=present, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(
        y_true_a, y_pred_a, labels=present, average="weighted", zero_division=0)

    print(f"\n{split} accuracy: {acc:.4f}")
    print(f"macro    P/R/F1: {p:.4f} / {r:.4f} / {f1:.4f}")
    print(f"weighted P/R/F1: {pw:.4f} / {rw:.4f} / {f1w:.4f}\n")
    target_names = [class_names[i] for i in present]
    print(classification_report(y_true_a, y_pred_a, labels=present,
                                target_names=target_names, zero_division=0))

    out_dir = ckpt_path.parent
    _plot_confusion(y_true_a, y_pred_a, present, target_names,
                    out_dir / f"confusion_{split}.png")

    metrics = {
        "split": split, "accuracy": acc,
        "macro": {"precision": p, "recall": r, "f1": f1},
        "weighted": {"precision": pw, "recall": rw, "f1": f1w},
        "num_samples": int(len(y_true)),
        "classes_present": target_names,
    }
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics + confusion matrix to {out_dir}")
    return metrics


def _plot_confusion(y_true, y_pred, labels, names, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(all="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6),
                                    max(5, len(names) * 0.6)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--ckpt", default="runs/stage1/best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    evaluate(cfg, args.ckpt, args.split)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
