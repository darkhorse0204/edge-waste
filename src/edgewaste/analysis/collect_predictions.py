# collect_predictions.py - one pass per split: classifier outputs, embeddings, frozen imagenet and hand-crafted features
"""Decode every image of a split exactly once and cache everything the
analyses need, so PCA/LDA/SVM, fit diagnostics, calibration and metrics all
run on arrays in seconds instead of re-reading ~28k images each time.

Per image it stores:
  * logits / softmax probabilities of the fine-tuned hybrid classifier;
  * the fused 1024-d embedding feeding the classifier head, and the two
    backbone attention weights (how much the model leaned on ConvNeXt vs ViT);
  * 768-d features of an ImageNet-pretrained ConvNeXt-Tiny that never saw this
    dataset — the "transfer learning without fine-tuning" baseline;
  * 1860-d hand-crafted features (96-bin HSV colour histogram + 1764-d HOG) —
    the classical computer-vision baseline.

Images are read in eval mode (no augmentation), so train-split outputs measure
what the model learned, not how it copes with random crops. Inference runs in
fp32: fp16 autocast flipped one of 4232 test predictions, and the cached
numbers must reproduce the reported ones exactly.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from edgewaste.classification.evaluate_classifier import _load_model
from edgewaste.config import Config
from edgewaste.data.waste_image_dataset import IMAGENET_MEAN, IMAGENET_STD, WasteDataset

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)
_HOG = None


def handcrafted_features(x: torch.Tensor) -> np.ndarray:
    """HSV colour histogram (3 x 32 bins, L1-normalised) + HOG on a 64x64
    greyscale copy (8x8 cells, 16x16 blocks, 9 orientations = 1764 values)."""
    global _HOG
    if _HOG is None:  # created lazily: one per dataloader worker process
        _HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    img = ((x.permute(1, 2, 0).numpy() * _STD + _MEAN).clip(0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = np.concatenate([
        cv2.calcHist([hsv], [c], None, [32], [0, 180 if c == 0 else 256]).ravel()
        for c in range(3)])
    hist /= hist.sum() + 1e-9
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (64, 64))
    hog = _HOG.compute(gray).ravel()
    return np.concatenate([hist, hog]).astype(np.float32)


class _WithHandcrafted(Dataset):
    def __init__(self, base: WasteDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, y, torch.from_numpy(handcrafted_features(x))


@torch.no_grad()
def collect_split(cfg: Config, ckpt: str, manifest: str, split: str, out_path: Path,
                  batch_size: int = 32, num_workers: int = 2) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = _load_model(Path(ckpt), cfg, device)
    frozen = timm.create_model("convnext_tiny", pretrained=True, num_classes=0,
                               global_pool="avg").to(device).eval()
    base = WasteDataset(manifest, split, cfg.model.image_size, train=False)
    loader = DataLoader(_WithHandcrafted(base), batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)
    out = {k: [] for k in ("logits", "emb", "attn", "imagenet", "handcrafted", "labels")}
    for x, y, hc in tqdm(loader, desc=f"collect {split}"):
        x = x.to(device, non_blocking=True)
        f_cnx = model.proj_cnx(model.convnext(x))
        f_vit = model.proj_vit(model.vit(x))
        fused, attn = model.fusion([f_cnx, f_vit])
        logits = model.head(fused)
        feats = frozen(x)
        out["logits"].append(logits.float().cpu().numpy())
        out["emb"].append(fused.float().cpu().numpy().astype(np.float16))
        out["attn"].append(attn.float().cpu().numpy().astype(np.float16))
        out["imagenet"].append(feats.float().cpu().numpy().astype(np.float16))
        out["handcrafted"].append(hc.numpy().astype(np.float16))
        out["labels"].append(y.numpy().astype(np.int16))
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    arrays["paths"] = base.df["path"].to_numpy(dtype=str)
    arrays["class_names"] = np.array(class_names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path


def load_split(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    arrays = {k: d[k] for k in d.files}
    z = arrays["logits"] - arrays["logits"].max(1, keepdims=True)
    arrays["probs"] = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    arrays["class_names"] = [str(c) for c in arrays["class_names"]]
    return arrays
