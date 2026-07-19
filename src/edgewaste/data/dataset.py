"""PyTorch Dataset + transforms driven by the split manifest."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..taxonomy import NUM_CLASSES

# ImageNet statistics (both timm ConvNeXt and ViT are pretrained on these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Augmentation per docs/stage-1 Module 2: rotation, brightness, crop, blur."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.1),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class WasteDataset(Dataset):
    """Reads (path, label) rows for one split from the manifest CSV."""

    def __init__(self, manifest: str | Path, split: str, image_size: int,
                 train: bool | None = None):
        df = pd.read_csv(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split='{split}' in {manifest}")
        self.split = split
        is_train = (split == "train") if train is None else train
        self.transform = build_transforms(image_size, train=is_train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)
        y = int(row["label"])
        return x, y

    # --- helpers for the trainer ---
    def label_counts(self) -> Counter:
        return Counter(int(v) for v in self.df["label"].tolist())

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights over the full canonical class set.

        Empty classes (e.g. 'hazardous' before a source is added) get weight 0
        so they don't distort the loss.
        """
        counts = self.label_counts()
        weights = np.zeros(NUM_CLASSES, dtype=np.float32)
        total = sum(counts.values())
        for cls_idx, n in counts.items():
            weights[cls_idx] = total / (len(counts) * n)
        return torch.tensor(weights, dtype=torch.float32)

    def sampler_weights(self) -> list[float]:
        """Per-sample weights for a WeightedRandomSampler (balanced batches)."""
        counts = self.label_counts()
        return [1.0 / counts[int(v)] for v in self.df["label"].tolist()]
