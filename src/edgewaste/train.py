"""Train the ConvNeXt + ViT hybrid classifier (Stage 1).

Usage:
    edgewaste-train --config configs/stage1.yaml
    edgewaste-train --config configs/stage1.yaml --smoke   # tiny fast run

Writes checkpoints and a metrics history to ``train.output_dir``. The best
(val-accuracy) checkpoint is saved as ``best.pt`` with the class list embedded,
so inference/evaluation never depend on re-deriving the taxonomy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm

from .config import Config
from .data.dataset import WasteDataset
from .models import build_model
from .taxonomy import CLASS_NAMES, NUM_CLASSES
from .utils import pick_device, seed_everything, accuracy, count_parameters


def _make_loaders(cfg: Config, smoke: bool):
    train_ds = WasteDataset(cfg.data.manifest, "train", cfg.model.image_size)
    try:
        val_ds = WasteDataset(cfg.data.manifest, "val", cfg.model.image_size)
    except ValueError:
        val_ds = None

    if smoke:
        # Subsample for a fast end-to-end sanity check.
        n = min(64, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        if val_ds is not None:
            m = min(32, len(val_ds))
            val_ds = Subset(val_ds, list(range(m)))

    base_train = train_ds.dataset if isinstance(train_ds, Subset) else train_ds

    workers = 0 if smoke else cfg.train.num_workers
    if cfg.train.use_class_weights and not smoke:
        sampler_w = base_train.sampler_weights()
        sampler = WeightedRandomSampler(sampler_w, num_samples=len(sampler_w),
                                        replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
            num_workers=workers, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
    return train_loader, val_loader, base_train


def _evaluate(model, loader, device, criterion) -> tuple[float, float]:
    model.eval()
    total, loss_sum, correct = 0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(cfg: Config, smoke: bool = False) -> Path:
    seed_everything(cfg.data.seed)
    device = pick_device()
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    train_loader, val_loader, base_train = _make_loaders(cfg, smoke)
    present = sorted(base_train.label_counts().keys())
    print(f"Classes present in train: {len(present)}/{NUM_CLASSES}")

    model = build_model(cfg.model, num_classes=NUM_CLASSES).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    class_weights = None
    if cfg.train.use_class_weights and not smoke:
        class_weights = base_train.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=cfg.train.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    epochs = 1 if smoke else cfg.train.epochs
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.train.lr, epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=min(0.3, cfg.train.warmup_epochs / max(epochs, 1)))

    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best_val = -1.0
    best_path = out_dir / "best.pt"
    patience = cfg.train.early_stop_patience
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, run_acc, seen = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            bs = y.size(0)
            run_loss += loss.item() * bs
            run_acc += accuracy(logits, y) * bs
            seen += bs
            pbar.set_postfix(loss=f"{run_loss/seen:.3f}", acc=f"{run_acc/seen:.3f}")

        tr_loss, tr_acc = run_loss / seen, run_acc / seen
        rec = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
               "seconds": round(time.time() - t0, 1)}

        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, device, criterion)
            rec["val_loss"], rec["val_acc"] = val_loss, val_acc
            improved = val_acc > best_val
            print(f"epoch {epoch:2d} | train_loss {tr_loss:.3f} acc {tr_acc:.3f}"
                  f" | val_loss {val_loss:.3f} acc {val_acc:.3f}"
                  f" | {rec['seconds']}s{'  *' if improved else ''}")
            if improved:
                best_val = val_acc
                stale = 0
                _save_checkpoint(best_path, model, cfg, epoch, val_acc)
            else:
                stale += 1
                if patience and stale >= patience:
                    print(f"Early stopping (no val improvement in {patience} epochs).")
                    history.append(rec)
                    break
        else:
            print(f"epoch {epoch:2d} | train_loss {tr_loss:.3f} acc {tr_acc:.3f}"
                  f" | {rec['seconds']}s")
            _save_checkpoint(best_path, model, cfg, epoch, tr_acc)

        history.append(rec)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Always keep a final checkpoint too.
    _save_checkpoint(out_dir / "last.pt", model, cfg, epochs, best_val)
    print(f"\nBest val acc: {best_val:.3f}" if best_val >= 0 else "Done.")
    print(f"Checkpoints in {out_dir}")
    return best_path


def _save_checkpoint(path: Path, model, cfg: Config, epoch: int, metric: float):
    # Atomic write: serialize to a temp file, then replace. A failed/partial
    # write (e.g. disk full) never clobbers a previously-good checkpoint.
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save({
            "model_state": model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "model_cfg": cfg.model.__dict__,
            "epoch": epoch,
            "metric": metric,
        }, tmp)
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to write checkpoint {path} ({exc}). "
            f"Low disk space? Each checkpoint is ~200MB."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the Stage 1 hybrid classifier.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny 1-epoch run on a subset to validate the pipeline.")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    train(cfg, smoke=args.smoke)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
