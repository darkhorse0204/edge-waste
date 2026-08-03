"""Shared utilities: device selection, seeding, metrics."""

from __future__ import annotations

import random

import numpy as np
import torch


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon fallback, harmless elsewhere.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pad_box(box: tuple[int, int, int, int], width: int, height: int,
            pad_frac: float = 0.15) -> tuple[int, int, int, int]:
    """Expand an xyxy box by `pad_frac` on every side, clamped to the frame.

    A detector box hugs the object, so the classifier sees the item with its
    edges cut off and no surrounding context — and the exact amount cut off
    changes every frame as the box jitters, which changes the prediction.
    Padding gives the crop a stable margin. Expansion is relative to each side
    independently, so wide and tall boxes both grow proportionally.

    Clamping means a box already touching a frame edge simply grows less on
    that side; it never produces coordinates outside the image.
    """
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * pad_frac
    dy = (y2 - y1) * pad_frac
    px1 = max(0, int(round(x1 - dx)))
    py1 = max(0, int(round(y1 - dy)))
    px2 = min(width, int(round(x2 + dx)))
    py2 = min(height, int(round(y2 + dy)))
    # Degenerate result (possible only if the input box was already degenerate)
    # — fall back to the original box rather than returning an empty crop.
    if px2 <= px1 or py2 <= py1:
        return box
    return px1, py1, px2, py2
