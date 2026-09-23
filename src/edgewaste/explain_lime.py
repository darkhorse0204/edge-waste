"""LIME explainability for the material classifier — a second, independent
explanation method alongside Grad-CAM (explain.py).

The two methods answer different questions and neither substitutes for the
other. Grad-CAM asks the *model's own gradients* which pixels drove this
class's score, so it can only ever explain what the network's weights
actually respond to, and only on the ConvNeXt branch (a spatial-gradient
method has no equivalent hook into a ViT's attention). LIME asks a much
more model-agnostic question: perturb the *input* and see which superpixels
the prediction is sensitive to, treating the classifier purely as a
black box. That is slower and noisier, but it does not care that half the
model (ViT) is otherwise unexplained, and disagreement between the two
methods is itself informative — if Grad-CAM highlights the label but LIME's
superpixel importance is scattered elsewhere, that is a real signal the
ConvNeXt branch is not what is actually deciding the fused prediction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _make_predict_fn(model, tfm, device):
    """LIME needs a function: batch of HWC uint8 arrays -> class probabilities.

    The classifier expects normalised tensors from `tfm`, so each perturbed
    superpixel image LIME generates has to go through the exact same
    preprocessing the model was trained on, or the explanation would be
    attributing importance under a distribution shift LIME never intended.
    """
    was_training = model.training
    model.eval()

    @torch.no_grad()
    def predict(images: np.ndarray) -> np.ndarray:
        batch = torch.stack([tfm(Image.fromarray(img)) for img in images]).to(device)
        probs = F.softmax(model(batch), dim=1)
        return probs.cpu().numpy()

    model.train(was_training)
    return predict


def lime_explain_crop(
    model, tfm, crop: Image.Image, device, class_names: list[str],
    num_samples: int = 500, num_features: int = 8, top_label_only: bool = True,
):
    """Explain one crop. Returns (label_idx, label_name, explanation, overlay_rgb).

    `num_samples` trades explanation stability for speed — LIME fits a local
    surrogate model on this many perturbed copies of the crop, so too few
    gives a noisy, unstable explanation. 500 is enough for a single-item crop
    at typical resolutions without becoming the bottleneck of a demo.
    """
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    predict_fn = _make_predict_fn(model, tfm, device)
    rgb = np.array(crop.convert("RGB"))

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        rgb, predict_fn, top_labels=1 if top_label_only else len(class_names),
        num_samples=num_samples, hide_color=0,
    )
    label_idx = explanation.top_labels[0]

    overlay_img, mask = explanation.get_image_and_mask(
        label_idx, positive_only=True, num_features=num_features, hide_rest=False
    )
    overlay_rgb = (mark_boundaries(overlay_img / 255.0, mask) * 255).astype(np.uint8)

    return label_idx, class_names[label_idx], explanation, overlay_rgb


def lime_feature_weights(explanation, label_idx: int) -> list[tuple[int, float]]:
    """The raw (superpixel_id, weight) pairs LIME's local surrogate fit —
    the actual quantitative attribution behind the visual overlay, for
    logging or the report rather than just the picture."""
    return explanation.local_exp[label_idx]
