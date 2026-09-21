"""Grad-CAM explainability for the hybrid classifier's ConvNeXt branch.

An on-demand path, not a per-frame overlay: Grad-CAM requires gradients and
would tank the live frame rate if run on every frame. Bound to a keypress in
the live pipeline instead. Wraps the backward pass in `torch.enable_grad()`
explicitly because the surrounding pipeline runs under `@torch.no_grad`.

Known gap: this covers only the ConvNeXt branch. The ViT branch has no
per-pixel visual explanation (ViT attention-rollout would close this) — but
the attention weights logged alongside every prediction (see
`logging_utils.py`) at least show how much each branch contributed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def explain_crop(
    model, crop_tensor: torch.Tensor, class_idx: int, out_path: str | Path,
    original_crop: Image.Image | None = None,
) -> Path:
    """Save a Grad-CAM heatmap overlay for one classified crop.

    `crop_tensor` is the already-normalised [1, 3, H, W] model input.
    `original_crop` (unnormalised PIL image) is used as the overlay
    background if given, else the tensor is denormalised for display.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layer = model.convnext.stages[-1]

    with torch.enable_grad():
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=crop_tensor, targets=None)[0]

    if original_crop is not None:
        rgb = np.array(original_crop.resize((crop_tensor.shape[-1], crop_tensor.shape[-2]))) / 255.0
    else:
        img = crop_tensor[0].detach().cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    overlay = show_cam_on_image(rgb.astype(np.float32), grayscale_cam, use_rgb=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)
    return out_path
