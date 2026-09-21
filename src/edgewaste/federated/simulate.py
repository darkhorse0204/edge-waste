"""FedAvg simulation loop. See package docstring for the scope/caveats."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..config import Config
from ..data.dataset import WasteDataset
from ..models import build_model
from ..taxonomy import CLASS_NAMES, NUM_CLASSES
from ..utils import accuracy, pick_device, seed_everything


def partition_indices(
    labels: list[int], num_clients: int, iid: bool, seed: int = 42
) -> list[list[int]]:
    """Split dataset indices across `num_clients` virtual clients.

    IID: shuffle then split evenly — every client sees a representative
    label mix, the easy case.

    Non-IID: sort by label into `2 * num_clients` shards, then deal 2 shards
    to each client (the standard "shard" non-IID construction) — each
    client ends up dominated by 1-2 material classes, closer to what a real
    smart bin in one physical location would actually see.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx = np.arange(n)

    if iid:
        rng.shuffle(idx)
        return [list(a) for a in np.array_split(idx, num_clients)]

    order = np.argsort(np.array(labels), kind="stable")
    num_shards = 2 * num_clients
    shards = np.array_split(order, num_shards)
    shard_order = rng.permutation(num_shards)
    client_shards = np.array_split(shard_order, num_clients)
    return [
        list(np.concatenate([shards[s] for s in shard_ids])) if len(shard_ids) else []
        for shard_ids in client_shards
    ]


def local_train(
    model: nn.Module, loader: DataLoader, device, epochs: int, lr: float,
) -> dict:
    """Train `model` in place for `epochs` on one client's data; return its
    resulting state_dict (moved to CPU, ready to aggregate)."""
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fedavg(state_dicts: list[dict], client_sizes: list[int]) -> dict:
    """Sample-size-weighted average of client state_dicts."""
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]
    avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state_dicts[0].items()}
    for sd, w in zip(state_dicts, weights):
        for k in avg:
            avg[k] += sd[k].float() * w
    # Cast back to each param's original dtype (state_dict may hold ints/bools too).
    ref = state_dicts[0]
    return {k: v.to(ref[k].dtype) for k, v in avg.items()}


@torch.no_grad()
def _evaluate(model, loader, device, criterion) -> tuple[float, float]:
    model.eval()
    total, loss_sum, correct = 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def run_simulation(
    cfg: Config, num_clients: int = 4, num_rounds: int = 3, local_epochs: int = 1,
    iid: bool = False, smoke: bool = False, base_ckpt: str | None = None,
    output_dir: str = "runs/federated",
) -> Path:
    seed_everything(cfg.data.seed)
    device = pick_device()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  clients={num_clients}  rounds={num_rounds}  "
          f"local_epochs={local_epochs}  iid={iid}")

    train_ds = WasteDataset(cfg.data.manifest, "train", cfg.model.image_size)
    try:
        val_ds = WasteDataset(cfg.data.manifest, "val", cfg.model.image_size, train=False)
    except ValueError:
        val_ds = train_ds  # tiny/smoke manifests may not carve out a val split

    labels = train_ds.df["label"].tolist()
    if smoke:
        cap = min(len(train_ds), 32 * num_clients)
        labels = labels[:cap]
    client_indices = partition_indices(labels, num_clients, iid, seed=cfg.data.seed)
    for i, idxs in enumerate(client_indices):
        print(f"  client {i}: {len(idxs)} samples")

    global_model = build_model(cfg.model, num_classes=NUM_CLASSES).to(device)
    if base_ckpt:
        ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
        global_model.load_state_dict(ckpt["model_state"])
        print(f"Initialised global model from {base_ckpt}")

    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    history: list[dict] = []
    for rnd in range(1, num_rounds + 1):
        t0 = time.time()
        client_states, client_sizes = [], []
        for c, idxs in enumerate(client_indices):
            if not idxs:
                continue
            local_model = copy.deepcopy(global_model)
            loader = DataLoader(Subset(train_ds, idxs),
                                 batch_size=min(cfg.train.batch_size, max(1, len(idxs))),
                                 shuffle=True)
            state = local_train(local_model, loader, device, local_epochs, cfg.train.lr)
            client_states.append(state)
            client_sizes.append(len(idxs))

        global_model.load_state_dict(fedavg(client_states, client_sizes))
        val_loss, val_acc = _evaluate(global_model, val_loader, device, criterion)
        rec = {"round": rnd, "val_loss": val_loss, "val_acc": val_acc,
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"round {rnd:2d} | global val_loss {val_loss:.3f} acc {val_acc:.3f} "
              f"| {rec['seconds']}s")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    best_path = out_dir / "global_best.pt"
    tmp = best_path.with_suffix(".pt.tmp")
    torch.save({
        "model_state": global_model.state_dict(),
        "class_names": list(CLASS_NAMES),
        "model_cfg": cfg.model.__dict__,
        "rounds": num_rounds,
        "metric": history[-1]["val_acc"] if history else None,
    }, tmp)
    os.replace(tmp, best_path)
    print(f"\nGlobal model saved to {best_path}")
    return best_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FedAvg federated-learning simulation.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--num-clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--iid", action="store_true", help="IID split (default: non-IID shards).")
    ap.add_argument("--smoke", action="store_true", help="Tiny fast run to validate wiring.")
    ap.add_argument("--base-ckpt", default=None, help="Warm-start global model from this checkpoint.")
    ap.add_argument("--output-dir", default="runs/federated")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    run_simulation(cfg, args.num_clients, args.rounds, args.local_epochs,
                    iid=args.iid, smoke=args.smoke, base_ckpt=args.base_ckpt,
                    output_dir=args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
