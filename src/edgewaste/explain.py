"""Grad-CAM explainability for the classifier branch.

One Grad-CAM implementation, three consumers: the live second window, the
on-demand 'e' snapshot, and anything added later. `gradcam_overlay()` is the
single place a heatmap is ever produced.

Cost matters here. Measured on an RTX 3050 Ti: a Grad-CAM pass is ~88 ms per
crop, against ~30 ms for a plain classification and ~32 ms for a detector
pass. Running it every frame on every detection would roughly halve the frame
rate, so `LiveExplainer` computes at a fixed frame interval, for one object
only, and reuses the cached panel in between.

Two honesty constraints are built in rather than glossed over:

  * The heatmap is targeted at the *displayed* class. Since the COCO identity
    prior can shift the final answer away from the classifier's own argmax,
    explaining the argmax would answer a question nobody asked.
  * The caption states what the heatmap actually accounts for. When the
    material came from object identity the classifier did not decide anything,
    and when it came from the classifier the heatmap still only covers the
    ConvNeXt branch — the ViT branch carries a large share of the fused
    decision and Grad-CAM on `stages[-1]` says nothing about it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .recognize import FROM_CONSTRAINED, FROM_IDENTITY, FROM_MATERIAL
from .utils import pad_box

# --- panel styling ---------------------------------------------------------
_BG = (28, 28, 30)
_FG = (235, 235, 235)
_DIM = (150, 150, 155)
_RULE = (70, 70, 74)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _wrap(text: str, max_w: int, scale: float, thickness: int = 1) -> list[str]:
    """Greedy word-wrap measured in real rendered pixels.

    OpenCV has no text layout, so captions silently run off the panel edge
    unless they are measured and broken explicitly.
    """
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        (w, _), _ = cv2.getTextSize(trial, _FONT, scale, thickness)
        if w <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _label_on(canvas, text: str, x: int, y: int, scale: float = 0.45) -> None:
    """Small caption with a dark plate behind it, so it stays legible over a
    bright crop (a white background made the plain white text vanish)."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 1)
    cv2.rectangle(canvas, (x - 4, y - th - 6), (x + tw + 4, y + 5), _BG, -1)
    cv2.putText(canvas, text, (x, y), _FONT, scale, _FG, 1, cv2.LINE_AA)


def build_gradcam(classifier) -> GradCAM:
    """Grad-CAM bound to the ConvNeXt branch's final stage.

    The ViT branch has no spatial feature map of the same kind, so a
    convolutional Grad-CAM cannot be pointed at it; explaining the fused model
    fully would need attention rollout as a second method. Captions downstream
    say so rather than implying full coverage.
    """
    target_layers = [classifier.convnext.stages[-1]]
    return GradCAM(model=classifier, target_layers=target_layers)


def gradcam_overlay(cam: GradCAM, classifier, tfm, crop: Image.Image, device,
                     class_idx: int = -1, alpha: float = 0.5
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Heatmap overlay for one crop. Returns (rgb_crop, rgb_overlay), both uint8.

    `class_idx` targets a specific class; -1 falls back to the model's argmax.
    `alpha` is the heatmap's share of the blend (0 = original, 1 = pure heat).
    """
    x = tfm(crop).unsqueeze(0).to(device)
    resized = crop.resize((x.shape[-1], x.shape[-2]))
    rgb_float = np.array(resized).astype(np.float32) / 255.0

    targets = [ClassifierOutputTarget(class_idx)] if class_idx >= 0 else None
    with torch.enable_grad():  # the caller's loop is under @torch.no_grad
        grayscale = cam(input_tensor=x, targets=targets)[0]

    overlay = show_cam_on_image(rgb_float, grayscale, use_rgb=True,
                                image_weight=1.0 - alpha)
    return (rgb_float * 255).astype(np.uint8), overlay


# ---------------------------------------------------------------------------
# Caption text — what the heatmap does and does not account for
# ---------------------------------------------------------------------------
_CAPTIONS = {
    FROM_MATERIAL: "Classifier decided this. Heatmap explains it.",
    FROM_CONSTRAINED: "Object identity narrowed the options; the classifier "
                      "chose within them. Heatmap explains that choice.",
    FROM_IDENTITY: "Material came from the OBJECT IDENTITY, not the classifier. "
                   "Heatmap shows the classifier's view only.",
}


@dataclass
class _Cached:
    panel: np.ndarray
    frame_idx: int
    track_id: int
    class_name: str


class LiveExplainer:
    """Second-window Grad-CAM: throttled, cached, single-object.

    Recomputes when the interval elapses, or immediately when the explained
    object or its class changes — so a genuine switch is reflected at once
    while a steady object costs almost nothing.
    """

    def __init__(self, cam: GradCAM, classifier, tfm, device,
                 interval: int = 10, alpha: float = 0.5, tile: int = 300,
                 pad_frac: float = 0.15):
        self.cam = cam
        self.classifier = classifier
        self.tfm = tfm
        self.device = device
        self.interval = int(interval)
        self.alpha = float(alpha)
        self.tile = int(tile)
        self.pad_frac = float(pad_frac)
        self._cache: _Cached | None = None
        self.last_ms = 0.0

    # -- target selection ---------------------------------------------------
    @staticmethod
    def _pick(dets):
        """Explain the most confident *identified* detection.

        Preferring a known class over an `unknown` one keeps the window useful:
        a heatmap for a class the system declined to commit to explains little.
        """
        if not dets:
            return None
        known = [d for d in dets if not d.is_unknown]
        return max(known or dets, key=lambda d: d.cls_conf)

    def _needs_refresh(self, det, frame_idx: int) -> bool:
        c = self._cache
        if c is None:
            return True
        if det.track_id != c.track_id or det.class_name != c.class_name:
            return True  # different object, or it changed its mind
        return (frame_idx - c.frame_idx) >= self.interval

    # -- main entry ---------------------------------------------------------
    def update(self, image: Image.Image, dets, frame_idx: int) -> np.ndarray:
        """Return the BGR panel for this frame (cached unless a refresh is due)."""
        det = self._pick(dets)
        if det is None:
            self._cache = None
            return self._placeholder()

        if self._needs_refresh(det, frame_idx):
            t0 = time.perf_counter()
            w, h = image.size
            box = det.crop_box or pad_box(det.box, w, h, self.pad_frac)
            crop = image.crop(box)
            crop_rgb, overlay = gradcam_overlay(
                self.cam, self.classifier, self.tfm, crop, self.device,
                class_idx=det.class_idx, alpha=self.alpha)
            self.last_ms = (time.perf_counter() - t0) * 1000
            panel = self._compose(crop_rgb, overlay, det, frame_idx)
            self._cache = _Cached(panel, frame_idx, det.track_id, det.class_name)

        # Age is drawn on a copy so the cached panel stays reusable.
        return self._with_age(self._cache.panel, frame_idx - self._cache.frame_idx)

    # -- rendering ----------------------------------------------------------
    def _canvas(self, w: int, h: int) -> np.ndarray:
        c = np.zeros((h, w, 3), np.uint8)
        c[:] = _BG
        return c

    def _placeholder(self) -> np.ndarray:
        w = self.tile * 2 + 48
        canvas = self._canvas(w, self.tile + 150)
        cv2.putText(canvas, "Grad-CAM Explainability", (16, 34), _FONT, 0.7,
                    _FG, 2, cv2.LINE_AA)
        cv2.line(canvas, (16, 48), (w - 16, 48), _RULE, 1)
        cv2.putText(canvas, "waiting for a detection...", (16, self.tile // 2 + 60),
                    _FONT, 0.6, _DIM, 1, cv2.LINE_AA)
        return canvas

    def _compose(self, crop_rgb, overlay, det, frame_idx) -> np.ndarray:
        t, pad = self.tile, 16
        w = t * 2 + pad * 3
        header = 54
        # Caption height is derived from the wrapped text rather than assumed,
        # so a long "decided_by" explanation cannot overflow the panel.
        caption = _wrap(_CAPTIONS.get(det.decided_by, ""), w - 2 * pad, 0.46)
        attn_txt = ""
        if det.attn:
            a_c, a_v = det.attn
            attn_txt = (f"fusion attention: ConvNeXt {a_c:.2f} | ViT {a_v:.2f}"
                        f"  -  heatmap covers the ConvNeXt branch only")
        attn_lines = _wrap(attn_txt, w - 2 * pad, 0.44) if attn_txt else []
        caption_h = 18 * (len(caption) + len(attn_lines)) + 52
        canvas = self._canvas(w, header + t + caption_h + pad)

        # header: the same label the main window shows
        material = det.class_name.replace("_", " ").title()
        title = (f"{det.object_name.title()} > {material}"
                 if det.object_name else material)
        cv2.putText(canvas, title, (pad, 32), _FONT, 0.78, _FG, 2, cv2.LINE_AA)
        conf = f"{det.cls_conf * 100:.1f}%   [{det.decided_by}]"
        (tw, _), _ = cv2.getTextSize(conf, _FONT, 0.6, 1)
        cv2.putText(canvas, conf, (w - pad - tw, 32), _FONT, 0.6, _DIM, 1,
                    cv2.LINE_AA)
        cv2.line(canvas, (pad, header - 8), (w - pad, header - 8), _RULE, 1)

        # the two tiles: what the classifier saw, and where it looked
        left = cv2.cvtColor(cv2.resize(crop_rgb, (t, t)), cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(cv2.resize(overlay, (t, t)), cv2.COLOR_RGB2BGR)
        canvas[header:header + t, pad:pad + t] = left
        canvas[header:header + t, pad * 2 + t:pad * 2 + t * 2] = right
        for x0 in (pad, pad * 2 + t):
            cv2.rectangle(canvas, (x0, header), (x0 + t, header + t), _RULE, 1)
        _label_on(canvas, "detected crop", pad + 6, header + t - 10)
        _label_on(canvas, "Grad-CAM (ConvNeXt)", pad * 2 + t + 6, header + t - 10)

        y = header + t + 24
        for line in caption:
            cv2.putText(canvas, line, (pad, y), _FONT, 0.46, _FG, 1, cv2.LINE_AA)
            y += 18
        # Attention split — states how much of the fused model this heatmap
        # actually covers, instead of implying it explains the whole thing.
        for line in attn_lines:
            cv2.putText(canvas, line, (pad, y), _FONT, 0.44, _DIM, 1, cv2.LINE_AA)
            y += 18

        self._colourbar(canvas, pad, y + 6, t, 10)
        cv2.putText(canvas, f"{self.last_ms:.0f} ms", (pad + t + 14, y + 15),
                    _FONT, 0.42, _DIM, 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _colourbar(canvas, x, y, width, height) -> None:
        ramp = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        bar = cv2.applyColorMap(np.repeat(ramp, height, axis=0), cv2.COLORMAP_JET)
        canvas[y:y + height, x:x + width] = bar
        cv2.putText(canvas, "low", (x, y + height + 13), _FONT, 0.38, _DIM, 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, "high", (x + width - 26, y + height + 13), _FONT,
                    0.38, _DIM, 1, cv2.LINE_AA)

    def _with_age(self, panel: np.ndarray, age: int) -> np.ndarray:
        out = panel.copy()
        txt = "live" if age == 0 else f"cached ({age}f)"
        (tw, _), _ = cv2.getTextSize(txt, _FONT, 0.42, 1)
        cv2.putText(out, txt, (out.shape[1] - tw - 16, out.shape[0] - 14),
                    _FONT, 0.42, _DIM, 1, cv2.LINE_AA)
        return out


def explain_frame(detector, classifier, tfm, class_names: list[str],
                   image: Image.Image, device, cam: GradCAM,
                   conf_threshold: float = 0.45,
                   out_dir: str = "runs/explanations",
                   pad_frac: float = 0.15) -> None:
    """On-demand 'e' snapshot. Same heatmap code as the live window."""
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    results = detector.predict(image, conf=conf_threshold, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print("No items detected — nothing to explain.")
        return

    w, h = image.size
    ts = int(time.time())
    for i, xyxy in enumerate(results[0].boxes.xyxy.tolist()):
        x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
        x2, y2 = min(w, int(xyxy[2])), min(h, int(xyxy[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop(pad_box((x1, y1, x2, y2), w, h, pad_frac))
        x = tfm(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, attn = classifier(x, return_attn=True)
        cls_conf, idx = torch.softmax(logits, dim=1)[0].max(0)

        _, overlay = gradcam_overlay(cam, classifier, tfm, crop, device,
                                      class_idx=int(idx))
        out_path = out_path_dir / f"explain_{ts}_item{i}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved {out_path} | class={class_names[int(idx)]} "
              f"conf={float(cls_conf):.2f} "
              f"attn(convnext,vit)=({attn[0,0]:.2f},{attn[0,1]:.2f})")
