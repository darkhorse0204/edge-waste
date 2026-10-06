# waste_image_dataset.py - pytorch dataset over the split manifest, with augmentation and class weights
"""PyTorch Dataset + transforms driven by the split manifest."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from edgewaste.taxonomy import NUM_CLASSES

# public waste datasets routinely ship a few truncated jpegs. decoding what is
# present beats aborting an epoch over the missing tail bytes.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# imagenet statistics (both timm convnext and vit are pretrained on these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Augmentation per docs/planning/stage_1 Module 2: rotation, brightness, crop, blur."""
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

    # how many neighbouring samples to try before declaring the split broken.
    _MAX_SUBSTITUTIONS = 50

    def __init__(self, manifest: str | Path, split: str, image_size: int,
                 train: bool | None = None):
        df = pd.read_csv(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split='{split}' in {manifest}")
        self.split = split
        is_train = (split == "train") if train is None else train
        self.transform = build_transforms(image_size, train=is_train)
        self._unreadable: set[str] = set()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        # a single unreadable file must never kill a multi-hour training run.
        # public dataset mirrors ship the occasional corrupt image, and losing
        # four finished epochs to one of them is a far worse outcome than
        # quietly training on a neighbouring sample instead. each bad path is
        # reported once so the corruption stays visible rather than silent.
        n = len(self.df)
        for offset in range(min(n, self._MAX_SUBSTITUTIONS)):
            row = self.df.iloc[(i + offset) % n]
            path = row["path"]
            try:
                img = Image.open(path).convert("RGB")
            except Exception as exc:  # unreadable, truncated, or not an image
                if path not in self._unreadable:
                    self._unreadable.add(path)
                    print(f"[WasteDataset] unreadable image skipped: {path} "
                          f"({type(exc).__name__}: {exc})")
                continue
            return self.transform(img), int(row["label"])
        raise RuntimeError(
            f"{self._MAX_SUBSTITUTIONS} consecutive unreadable images from index "
            f"{i} in split '{self.split}'. The dataset is likely corrupt or the "
            f"paths in the manifest are stale — re-run edgewaste-ingest."
        )

    # --- helpers for the trainer ---
    def label_counts(self) -> Counter:
        return Counter(int(v) for v in self.df["label"].tolist())

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights over the full canonical class set.

        Any class with zero training samples gets weight 0 so it doesn't
        distort the loss (defensive — all 7 classes should be populated once
        `edgewaste-ingest` has run against all three declared sources).
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
