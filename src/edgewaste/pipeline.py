"""Two-stage inference: YOLO26 localizes items, the Stage 1 classifier types
each crop.

This is the tech-stack doc's "detect-then-classify" pipeline (section 2.3):
swapping/upgrading either stage independently is why detection (YOLO26,
ultralytics) and classification (ConvNeXt+ViT hybrid, timm/torch) stay in
separate modules that only meet here.

Live-demo stability (all inference-side, no retraining, checkpoints untouched):
  * detections below `detect.conf_threshold` never reach the classifier;
  * each box is padded before cropping so the classifier sees the whole item
    plus a stable margin (`infer.pad_frac`);
  * per-object rolling-window smoothing over softmax outputs replaces the
    per-frame argmax (`infer.smoothing`, `infer.history` — see stabilize.py);
  * a smoothed confidence below `infer.cls_conf_threshold` displays "unknown"
    rather than a class the model doesn't actually believe in;
  * box colour encodes confidence: green >= 80%, yellow >= 60%, red below.

Usage:
    edgewaste-pipeline --det-ckpt runs/detect/taco_single_class/weights/best.pt \
        --cls-ckpt runs/stage1/best.pt path/to/image_or_dir
    edgewaste-pipeline --det-ckpt ... --cls-ckpt ... --camera
    edgewaste-pipeline --det-ckpt ... --cls-ckpt ... --camera --log-csv runs/predictions_log.csv

Live camera demo keys:
    q  -  quit
    e  -  save a Grad-CAM + attention-weight explanation for the current frame
          (a separate, slower on-demand path — see explain.py for why this
          can't run inside the main per-frame loop)
    r  -  reset the smoothing history (use after swapping the object)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import Config
from .infer import _iter_paths, load_for_inference
from .utils import pad_box, pick_device

from .explain import LiveExplainer, build_gradcam, explain_frame
from .logging_utils import PredictionLogger
from .recognize import (FROM_MATERIAL, ObjectRecognizer, apply_identity_prior,
                         match_to_boxes)
from .stabilize import PredictionStabilizer, iou
from .taxonomy import NAME_TO_INDEX


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords
    det_conf: float
    class_name: str  # what is displayed — may be the unknown label
    cls_conf: float  # what is displayed — smoothed when smoothing is on
    attn: tuple[float, float] | None = None  # (convnext_weight, vit_weight)
    # Pre-smoothing values for this frame alone. Kept so the CSV records both
    # and the flicker reduction can be quantified after the fact.
    raw_class_name: str = ""
    raw_cls_conf: float = 0.0
    track_id: int = -1
    votes: int = 0  # frames in the window agreeing with the displayed class
    window: int = 1  # frames currently in the window
    is_unknown: bool = False
    # Object identity (COCO). Empty when nothing was recognised.
    object_name: str = ""
    object_conf: float = 0.0
    # Which stage actually decided the material — identity / constrained /
    # material. Displayed and logged so a prediction is always traceable.
    decided_by: str = FROM_MATERIAL
    # The padded box actually fed to the classifier, and the displayed class's
    # index. Both exist so the Grad-CAM window can re-derive the exact same
    # input and explain the class being shown rather than the raw argmax.
    crop_box: tuple[int, int, int, int] | None = None
    class_idx: int = -1


def load_detector(det_ckpt: str, device):
    from ultralytics import YOLO
    model = YOLO(det_ckpt)
    return model

def _filter_contained_boxes(boxes, containment_thresh: float = 0.8) -> list[bool]:
    """Drop boxes that sit mostly inside a larger box (e.g. a bottle-cap box
    nested inside the whole-bottle box). Standard IoU-based NMS misses this:
    a small box fully inside a big one has low IoU with it (IoU divides by
    the union, which the big box dominates), so it survives NMS untouched."""
    n = len(boxes)
    keep = [True] * n
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    for i in range(n):
        for j in range(n):
            if i == j or not keep[i] or not keep[j] or areas[i] >= areas[j]:
                continue
            xi1, yi1, xi2, yi2, _ = boxes[i]
            xj1, yj1, xj2, yj2, _ = boxes[j]
            ix1, iy1 = max(xi1, xj1), max(yi1, yj1)
            ix2, iy2 = min(xi2, xj2), min(yi2, yj2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if areas[i] > 0 and inter / areas[i] > containment_thresh:
                keep[i] = False  # i is mostly inside a bigger box j — drop it
    return keep

@torch.no_grad()
def run_frame(
    detector, classifier, tfm, class_names: list[str], image: Image.Image,
    device, conf_threshold: float = 0.45, topk: int = 1,
    containment_thresh: float = 0.8, pad_frac: float = 0.15,
    cls_conf_threshold: float = 0.60, unknown_label: str = "unknown",
    stabilizer: PredictionStabilizer | None = None, frame_idx: int = 0,
    recognizer: ObjectRecognizer | None = None,
    recog_iou: float = 0.45, add_unmatched: bool = True,
    certainty: float = 0.95,
) -> list[Detection]:
    """Detect items in a PIL image, then classify each crop.

    `stabilizer=None` classifies each detection independently — correct for
    still images, where consecutive inputs are unrelated. The camera loop
    passes a stabilizer so predictions carry across frames.

    `recognizer` supplies COCO object identity, folded in as a prior over the
    material distribution (see recognize.py). None reverts to material-only.
    """
    dets: list[Detection] = []
    w, h = image.size

    results = detector.predict(image, conf=conf_threshold, verbose=False)
    raw_boxes = []
    if results:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for xyxy, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                x1, y1, x2, y2 = (max(0, int(xyxy[0])), max(0, int(xyxy[1])),
                                  min(w, int(xyxy[2])), min(h, int(xyxy[3])))
                if x2 <= x1 or y2 <= y1:
                    continue
                raw_boxes.append((x1, y1, x2, y2, conf))

    # Object identity, and the boxes it contributes on its own.
    matches = recognizer.detect(image) if recognizer is not None else []
    if recognizer is not None and add_unmatched:
        # A confidently-recognised waste object the TACO detector missed is
        # still a real item — TACO is litter-tuned and routinely fails on tidy,
        # well-lit objects. Only *mapped* classes are added; a recognised
        # person or chair contributes nothing rather than being force-fitted
        # into the taxonomy.
        for m in matches:
            if not m.is_waste:
                continue
            already = any(iou(m.box, (b[0], b[1], b[2], b[3])) >= recog_iou
                          for b in raw_boxes)
            if not already:
                raw_boxes.append((*m.box, m.conf))

    if not raw_boxes:
        return dets

    keep = _filter_contained_boxes(raw_boxes, containment_thresh)

    kept_boxes: list[tuple[int, int, int, int]] = []
    kept_det_conf: list[float] = []
    kept_probs: list[np.ndarray] = []
    kept_attn: list[tuple[float, float]] = []
    kept_object: list[tuple[str, float, str]] = []  # (name, conf, decided_by)
    kept_crop_box: list[tuple[int, int, int, int]] = []

    for (x1, y1, x2, y2, conf), keep_flag in zip(raw_boxes, keep):
        if not keep_flag:
            continue
        # Classify a padded crop, but keep the tight box for drawing and for
        # track association — the padded region is context for the model, not
        # a claim about where the object is.
        crop_box = pad_box((x1, y1, x2, y2), w, h, pad_frac)
        crop = image.crop(crop_box)
        x = tfm(crop).unsqueeze(0).to(device)
        logits, attn = classifier(x, return_attn=True)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        obj_name, obj_conf, decided_by = "", 0.0, FROM_MATERIAL
        if matches:
            match = match_to_boxes((x1, y1, x2, y2), matches, recog_iou)
            if match is not None and match.is_waste:
                probs, decided_by = apply_identity_prior(
                    probs, match.rule, match.conf, NAME_TO_INDEX, certainty)
                obj_name, obj_conf = match.label, match.conf

        kept_boxes.append((x1, y1, x2, y2))
        kept_det_conf.append(float(conf))
        kept_probs.append(probs)
        kept_attn.append(tuple(attn[0].tolist()))
        kept_object.append((obj_name, obj_conf, decided_by))
        kept_crop_box.append(crop_box)

    if not kept_boxes:
        return dets

    if stabilizer is not None:
        smoothed = stabilizer.update(frame_idx, kept_boxes, kept_probs)
    else:
        smoothed = None

    for i, (box, det_conf, probs, attn, obj, crop_box) in enumerate(
            zip(kept_boxes, kept_det_conf, kept_probs, kept_attn, kept_object,
                kept_crop_box)):
        raw_idx = int(probs.argmax())
        raw_conf = float(probs[raw_idx])
        obj_name, obj_conf, decided_by = obj

        if smoothed is not None:
            res = smoothed[i]
            is_unknown = res.class_idx is None
            shown_idx = raw_idx if res.class_idx is None else res.class_idx
            shown_name = unknown_label if is_unknown else class_names[res.class_idx]
            shown_conf = res.confidence
            track_id, votes, window = res.track_id, res.votes, res.window
        else:
            is_unknown = raw_conf < cls_conf_threshold
            shown_idx = raw_idx
            shown_name = unknown_label if is_unknown else class_names[raw_idx]
            shown_conf = raw_conf
            track_id, votes, window = -1, 1, 1

        dets.append(Detection(
            box=box, det_conf=det_conf,
            class_name=shown_name, cls_conf=shown_conf,
            attn=attn,
            raw_class_name=class_names[raw_idx], raw_cls_conf=raw_conf,
            track_id=track_id, votes=votes, window=window,
            is_unknown=is_unknown,
            object_name=obj_name, object_conf=obj_conf, decided_by=decided_by,
            crop_box=crop_box, class_idx=shown_idx,
        ))
    return dets


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)
_RED = (0, 0, 255)

MAIN_WINDOW = "Edge-Waste Detection"
GRADCAM_WINDOW = "Grad-CAM Explainability"


def _colour_for(det: Detection, high: float, mid: float):
    """BGR box colour from the displayed confidence."""
    if det.is_unknown:
        return _RED
    if det.cls_conf >= high:
        return _GREEN
    if det.cls_conf >= mid:
        return _YELLOW
    return _RED


def _draw_label(frame, x: int, y: int, lines: list[str], colour) -> None:
    """Filled label block anchored above (x, y), flipped below if it won't fit."""
    import cv2

    font, scale, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2, 6
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    line_h = max(s[1] for s in sizes) + 8
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = line_h * len(lines) + pad

    y0 = y - box_h
    if y0 < 0:  # no room above the box — draw just below its top edge instead
        y0 = y
    x0 = max(0, min(x, frame.shape[1] - box_w))

    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), colour, -1)
    # Black text on the bright fills, white on red — keeps contrast readable.
    b, g, r = colour
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    text_colour = (0, 0, 0) if luma > 140 else (255, 255, 255)
    for i, line in enumerate(lines):
        ty = y0 + pad + line_h * i + sizes[i][1]
        cv2.putText(frame, line, (x0 + pad, ty), font, scale, text_colour,
                    thick, cv2.LINE_AA)


def _draw_detection(frame, det: Detection, high: float, mid: float) -> None:
    import cv2

    colour = _colour_for(det, high, mid)
    x1, y1, x2, y2 = det.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    material = det.class_name.replace("_", " ").title()
    # "Banana > Organic" makes the reasoning legible at a glance: what the
    # system thinks the object is, and what it concluded about the material.
    top = f"{det.object_name.title()} > {material}" if det.object_name else material
    lines = [top, f"{det.cls_conf * 100:.1f}%  [{det.decided_by}]"]
    _draw_label(frame, x1, y1 - 4, lines, colour)


def _draw_hud(frame, text: str) -> None:
    import cv2

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - th - 14), (tw + 16, h), (32, 32, 32), -1)
    cv2.putText(frame, text, (8, h - 8), font, scale, (220, 220, 220), thick,
                cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_images(cfg: Config, det_ckpt: str, cls_ckpt: str, inputs: list[str]):
    device = pick_device()
    classifier, class_names, tfm = load_for_inference(cls_ckpt, cfg, device)
    detector = load_detector(det_ckpt, device)
    conf = cfg.detect.conf_threshold
    recognizer = _build_recognizer(cfg)

    any_found = False
    for path in _iter_paths(inputs):
        any_found = True
        image = Image.open(path).convert("RGB")
        # No stabilizer: separate files are not a temporal sequence.
        dets = run_frame(detector, classifier, tfm, list(class_names), image,
                         device, conf_threshold=conf,
                         containment_thresh=cfg.infer.containment_thresh,
                         pad_frac=cfg.infer.pad_frac,
                         cls_conf_threshold=cfg.infer.cls_conf_threshold,
                         unknown_label=cfg.infer.unknown_label,
                         recognizer=recognizer,
                         recog_iou=cfg.recognize.iou_match,
                         add_unmatched=cfg.recognize.add_unmatched,
                         certainty=cfg.recognize.certainty)
        if not dets:
            print(f"{path.name:<40} -> no items detected (conf >= {conf})")
            continue
        for i, d in enumerate(dets):
            obj = f"{d.object_name}({d.object_conf*100:.0f}%) -> " if d.object_name else ""
            print(f"{path.name:<40} item {i}: {obj}{d.class_name:<10} "
                  f"cls={d.cls_conf*100:5.1f}%  det={d.det_conf*100:5.1f}%  "
                  f"via={d.decided_by}")
    if not any_found:
        print("No images found in the given inputs.")


def _build_recognizer(cfg: Config) -> ObjectRecognizer | None:
    """Load the COCO identity model, or explain why it was skipped.

    A missing/undownloadable COCO checkpoint must not take the whole demo down
    — the pipeline degrades to material-only, which is exactly the old
    behaviour, and says so.
    """
    if not cfg.recognize.enabled:
        return None
    try:
        rec = ObjectRecognizer(cfg.recognize.model, cfg.recognize.conf_threshold)
        print(f"[identity] COCO recogniser '{cfg.recognize.model}' "
              f"({len(rec.names)} classes) conf>={cfg.recognize.conf_threshold:.2f}")
        return rec
    except Exception as exc:  # weights missing, no network, etc.
        print(f"[identity] DISABLED — could not load '{cfg.recognize.model}' ({exc}). "
              f"Falling back to material-only classification.")
        return None


def run_camera(cfg: Config, det_ckpt: str, cls_ckpt: str, cam_index: int,
                log_csv: str | None = None, explain_dir: str = "runs/explanations"):
    import cv2

    device = pick_device()
    classifier, class_names, tfm = load_for_inference(cls_ckpt, cfg, device)
    detector = load_detector(det_ckpt, device)
    ic = cfg.infer
    conf = cfg.detect.conf_threshold
    recognizer = _build_recognizer(cfg)
    cam = build_gradcam(classifier)
    logger = PredictionLogger(log_csv) if log_csv else None
    print(f"[logging] {'writing to ' + str(logger.log_path) if logger else 'DISABLED — log_csv is None'}")

    stabilizer = None
    if ic.smoothing != "none":
        stabilizer = PredictionStabilizer(
            history=ic.history, mode=ic.smoothing,
            conf_threshold=ic.cls_conf_threshold,
            switch_margin=ic.switch_margin,
            iou_match=ic.iou_match, max_age=ic.max_age)
    print(f"[stability] det>={conf:.2f}  cls>={ic.cls_conf_threshold:.2f}  "
          f"pad={ic.pad_frac:.0%}  smoothing={ic.smoothing}"
          f"{'' if stabilizer is None else f'({ic.history} frames)'}")

    ec = cfg.explain
    explainer = LiveExplainer(cam, classifier, tfm, device,
                              interval=ec.interval, alpha=ec.alpha,
                              tile=ec.tile, pad_frac=ic.pad_frac)
    gradcam_on = ec.enabled
    print(f"[grad-cam] live window {'ON' if gradcam_on else 'OFF'} "
          f"(toggle with 'g') — refresh every {ec.interval} frames, "
          f"1 object, alpha={ec.alpha}")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {cam_index}.")
    print("Detect+classify demo running. Keys: q quit | g live Grad-CAM | "
          "e save explanation | r reset smoothing")
    frame_idx, fps = 0, 0.0
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            dets = run_frame(detector, classifier, tfm, list(class_names),
                             image, device, conf_threshold=conf,
                             containment_thresh=ic.containment_thresh,
                             pad_frac=ic.pad_frac,
                             cls_conf_threshold=ic.cls_conf_threshold,
                             unknown_label=ic.unknown_label,
                             stabilizer=stabilizer, frame_idx=frame_idx,
                             recognizer=recognizer,
                             recog_iou=cfg.recognize.iou_match,
                             add_unmatched=cfg.recognize.add_unmatched,
                             certainty=cfg.recognize.certainty)
            for d in dets:
                if logger:
                    logger.log(d)
                _draw_detection(frame, d, ic.colour_high, ic.colour_mid)

            # Second window. Built from the same `dets` produced above, so the
            # two windows can never describe different frames — the heatmap may
            # be a few frames old (and says so), but it always belongs to a
            # detection that really happened, with the label it really had.
            if gradcam_on:
                cv2.imshow(GRADCAM_WINDOW,
                           explainer.update(image, dets, frame_idx))

            dt = time.time() - t0
            fps = (0.9 * fps + 0.1 / dt) if dt > 0 and fps else (1 / dt if dt > 0 else 0.0)
            if ic.show_hud:
                _draw_hud(frame, f"FPS {fps:4.1f} | det>={conf:.2f} "
                                 f"| cls>={ic.cls_conf_threshold:.2f} "
                                 f"| smooth={ic.smoothing}"
                                 f"{'' if stabilizer is None else f'/{ic.history}'} "
                                 f"| pad={ic.pad_frac:.0%} | items={len(dets)}"
                                 f"{' | GradCAM' if gradcam_on else ''}")

            cv2.imshow(MAIN_WINDOW, frame)
            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("g"):
                gradcam_on = not gradcam_on
                if not gradcam_on:
                    try:
                        cv2.destroyWindow(GRADCAM_WINDOW)
                    except cv2.error:
                        pass  # never created yet — nothing to close
                print(f"[grad-cam] live window {'ON' if gradcam_on else 'OFF'}")
            if key == ord("r") and stabilizer is not None:
                stabilizer.reset()
                print("[stability] smoothing history cleared.")
            if key == ord("e"):
                explain_frame(detector, classifier, tfm, list(class_names),
                              image, device, cam, conf_threshold=conf,
                              out_dir=explain_dir, pad_frac=ic.pad_frac)
            if cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if logger:
            logger.close()
        cap.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Detect-then-classify pipeline demo.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--det-ckpt", required=True,
                     help="YOLO26 detector weights, e.g. runs/detect/taco_single_class/weights/best.pt")
    ap.add_argument("--cls-ckpt", default="runs/stage1/best.pt")
    ap.add_argument("--camera", action="store_true", help="Live webcam demo.")
    ap.add_argument("--cam-index", type=int, default=0)
    ap.add_argument("--log-csv", default=None,
                     help="Path to write per-detection CSV log (camera mode only).")
    ap.add_argument("--explain-dir", default="runs/explanations",
                     help="Directory for Grad-CAM explanation snapshots (camera mode only).")
    # Stability knobs — override the config without editing YAML, so they can
    # be swept live during a demo.
    ap.add_argument("--det-conf", type=float, default=None,
                     help="Detector confidence gate (default: config, 0.45).")
    ap.add_argument("--cls-conf", type=float, default=None,
                     help="Classifier confidence gate below which 'unknown' is shown (default 0.60).")
    ap.add_argument("--pad", type=float, default=None,
                     help="Crop padding as a fraction of box size (default 0.15).")
    ap.add_argument("--smoothing", choices=["mean", "vote", "none"], default=None,
                     help="Temporal smoothing mode (default 'mean').")
    ap.add_argument("--history", type=int, default=None,
                     help="Frames of prediction history per object (default 10).")
    ap.add_argument("--switch-margin", type=float, default=None,
                     help="Hysteresis margin before the label may change (default 0.05).")
    ap.add_argument("--no-hud", action="store_true", help="Hide the on-screen settings overlay.")
    ap.add_argument("--no-identity", action="store_true",
                     help="Disable the COCO object recogniser (material-only, as before).")
    ap.add_argument("--identity-conf", type=float, default=None,
                     help="COCO recogniser confidence gate (default 0.50).")
    ap.add_argument("--gradcam", action="store_true",
                     help="Open the live Grad-CAM window at startup (toggle with 'g').")
    ap.add_argument("--gradcam-interval", type=int, default=None,
                     help="Frames between Grad-CAM recomputes (default 10).")
    ap.add_argument("--gradcam-alpha", type=float, default=None,
                     help="Heatmap opacity, 0-1 (default 0.5).")
    ap.add_argument("--gradcam-tile", type=int, default=None,
                     help="Pixel size of each panel tile (default 300).")
    ap.add_argument("inputs", nargs="*", help="Image files or directories.")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)

    if args.det_conf is not None:
        cfg.detect.conf_threshold = args.det_conf
    if args.cls_conf is not None:
        cfg.infer.cls_conf_threshold = args.cls_conf
    if args.pad is not None:
        cfg.infer.pad_frac = args.pad
    if args.smoothing is not None:
        cfg.infer.smoothing = args.smoothing
    if args.history is not None:
        cfg.infer.history = args.history
    if args.switch_margin is not None:
        cfg.infer.switch_margin = args.switch_margin
    if args.no_hud:
        cfg.infer.show_hud = False
    if args.no_identity:
        cfg.recognize.enabled = False
    if args.identity_conf is not None:
        cfg.recognize.conf_threshold = args.identity_conf
    if args.gradcam:
        cfg.explain.enabled = True
    if args.gradcam_interval is not None:
        cfg.explain.interval = args.gradcam_interval
    if args.gradcam_alpha is not None:
        cfg.explain.alpha = args.gradcam_alpha
    if args.gradcam_tile is not None:
        cfg.explain.tile = args.gradcam_tile

    if args.camera:
        run_camera(cfg, args.det_ckpt, args.cls_ckpt, args.cam_index,
                   log_csv=args.log_csv, explain_dir=args.explain_dir)
    elif args.inputs:
        run_images(cfg, args.det_ckpt, args.cls_ckpt, args.inputs)
    else:
        ap.error("Provide image paths/dirs, or use --camera.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())