"""Pixel-level segmentation via SAM (Segment Anything), box-prompted from the
existing detector's output.

The detector gives a rectangular box; a rectangle around a bottle also
contains a slice of whatever background is behind it, and that background
pixel content leaks into the classifier's crop as noise it never asked for.
Box-prompted SAM turns that rectangle into a pixel-accurate mask of the
actual object, so the classifier can be fed a background-suppressed crop
instead. Whether that measurably helps classification (vs. hurting by
removing context the ConvNeXt branch also uses) is an open, testable
question — `segment_and_mask` exposes both the raw crop and the
mask-suppressed crop for exactly that comparison, rather than assuming the
answer.

MobileSAM, not full SAM, for the same reason the detector is a YOLO26n and
not a YOLO26x: this project targets edge deployment, and MobileSAM is a
~40 MB distillation of SAM's image encoder built for exactly this budget,
against SAM ViT-H's ~2.4 GB. Box-prompted mode is used throughout rather
than SAM's automatic "segment everything" mode — everything-mode finds all
objects in a scene with no notion of *which* one the detector already
localised, which is a second detection problem this project doesn't need
solved twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class SegmentResult:
    mask: np.ndarray  # HxW bool, True where the item is
    mask_area_px: int
    box_area_px: int
    fill_ratio: float  # mask_area / box_area — how much of the box the object actually fills


def load_sam(model_path: str = "mobile_sam.pt"):
    from ultralytics import SAM
    return SAM(model_path)


def segment_box(sam_model, image: Image.Image, box: tuple[int, int, int, int]) -> SegmentResult:
    """Box-prompted segmentation for one detection.

    A single item, single box call — SAM's box prompt is unambiguous here
    (there is exactly one object the caller means), so no confidence
    threshold or NMS is needed the way the detector itself requires it.
    """
    import cv2

    rgb = np.array(image.convert("RGB"))
    result = sam_model(rgb, bboxes=[list(box)], verbose=False)[0]

    x1, y1, x2, y2 = box
    box_area = max(0, x2 - x1) * max(0, y2 - y1)

    if result.masks is None or len(result.masks.data) == 0:
        # SAM found nothing for this box (degenerate box, or a genuinely
        # empty region) — fall back to the box itself as a "mask" so callers
        # never have to special-case a missing result.
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[max(0, y1):y2, max(0, x1):x2] = True
    else:
        mask_t = result.masks.data[0].cpu().numpy()
        mask = cv2.resize(mask_t.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)

    mask_area = int(mask.sum())
    return SegmentResult(
        mask=mask, mask_area_px=mask_area, box_area_px=box_area,
        fill_ratio=(mask_area / box_area) if box_area > 0 else 0.0,
    )


def segment_and_mask(
    sam_model, image: Image.Image, box: tuple[int, int, int, int],
    background: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, Image.Image, SegmentResult]:
    """Return (raw_crop, mask_suppressed_crop, segmentation_result) for one box.

    The suppressed crop replaces every pixel outside the mask with a flat
    grey (YOLO's own standard padding colour, chosen so the classifier — a
    timm/ImageNet-family model — sees a neutral fill it has already
    encountered during its own pretraining's letterboxing, not a colour that
    itself looks like a material).
    """
    seg = segment_box(sam_model, image, box)
    x1, y1, x2, y2 = box
    raw_crop = image.crop(box)

    rgb = np.array(image.convert("RGB"))
    suppressed = np.full_like(rgb, background)
    suppressed[seg.mask] = rgb[seg.mask]
    suppressed_crop = Image.fromarray(suppressed).crop(box)

    return raw_crop, suppressed_crop, seg


def overlay_mask(image: Image.Image, seg: SegmentResult,
                  color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    """RGB visualisation: the mask tinted over the original image."""
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    overlay = rgb.copy()
    overlay[seg.mask] = (1 - alpha) * rgb[seg.mask] + alpha * np.array(color, dtype=np.float32)
    return overlay.astype(np.uint8)
