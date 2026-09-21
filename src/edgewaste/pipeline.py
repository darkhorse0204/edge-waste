"""Full detect-then-classify-then-decide pipeline: YOLO26 localizes items,
the ConvNeXt+ViT hybrid classifies each crop with MC-Dropout uncertainty,
simulated sensors feed the OCI contamination score, and the decision engine
routes each item to a conveyor gate — matching the architecture in the
Review-1 report's Fig 9.1. Every detection is logged to CSV.

Grad-CAM heatmaps are generated on-demand only (the 'e' keypress in camera
mode), since Grad-CAM needs gradients and running it every frame would tank
the live frame rate.

Usage:
    edgewaste-pipeline --det-ckpt runs/detect/taco_single_class/weights/best.pt \
        --cls-ckpt runs/stage1/best.pt path/to/image_or_dir
    edgewaste-pipeline --det-ckpt ... --cls-ckpt ... --camera
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import Config
from .confidence import mc_dropout_predict
from .decision import Decision, actuate_conveyor, decide
from .infer import IMAGE_EXTS, _iter_paths, load_for_inference
from .logging_utils import DetectionRecord, PredictionLogger
from .oci import compute_oci, fit_oci_weights
from .oci.normalize import normalize_gas, normalize_moisture
from .oci.synthetic import DEFAULT_ANCHORS, generate_synthetic_calibration_data
from .sensors import simulate_sensors
from .utils import pick_device


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords
    det_conf: float
    class_name: str
    cls_conf: float
    uncertainty: float = float("nan")
    attn_convnext: float = float("nan")
    attn_vit: float = float("nan")
    oci_score: float | None = None
    decision: Decision | None = None


def load_detector(det_ckpt: str, device):
    from ultralytics import YOLO
    model = YOLO(det_ckpt)
    return model


def _filter_contained_boxes(boxes: list[tuple], confs: list[float], thresh: float = 0.8):
    """Drop a box that is >= `thresh` contained inside another box.

    Standard IoU-based NMS misses this: a small box fully inside a large box
    has *low* IoU with it (IoU divides by the union, dominated by the large
    box), so a bottle-cap detection nested inside a whole-bottle detection
    survives NMS untouched and the pipeline would classify the same physical
    item twice. Containment (intersection / area(smaller)) catches it. O(n^2)
    pairwise scan — fine given a frame carries a handful of detections.
    """
    n = len(boxes)
    keep = [True] * n

    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def inter(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            smaller, larger = (i, j) if area(boxes[i]) <= area(boxes[j]) else (j, i)
            a_small = area(boxes[smaller])
            if a_small == 0:
                keep[smaller] = False
                continue
            ratio = inter(boxes[i], boxes[j]) / a_small
            if ratio >= thresh:
                keep[smaller] = False
    return [b for b, k in zip(boxes, keep) if k], [c for c, k in zip(confs, keep) if k]


def _fit_demo_oci_model():
    """Fit the OCI model once at startup on synthetic calibration data (no
    real calibration_data.csv exists yet — see oci/synthetic.py)."""
    df = generate_synthetic_calibration_data()
    f_m = df["moisture_raw"].apply(lambda v: normalize_moisture(v, DEFAULT_ANCHORS)).to_numpy()
    f_g = df["rs"].apply(lambda v: normalize_gas(v, DEFAULT_ANCHORS)).to_numpy()
    y = df["contaminated"].to_numpy()
    model, _ = fit_oci_weights(f_m, f_g, y)
    return model


@torch.no_grad()
def run_frame(
    detector, classifier, tfm, class_names: list[str], image: Image.Image,
    device, conf_threshold: float = 0.35, oci_model=None, mc_passes: int = 8,
    rng: np.random.Generator | None = None,
) -> list[Detection]:
    """Detect items, then classify + sense + decide for each surviving crop."""
    rng = rng or np.random.default_rng()
    results = detector.predict(image, conf=conf_threshold, verbose=False)
    if not results:
        return []
    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return []
    w, h = image.size
    raw_boxes, raw_confs = [], []
    for xyxy, conf in zip(boxes_obj.xyxy.tolist(), boxes_obj.conf.tolist()):
        x1, y1, x2, y2 = (max(0, int(xyxy[0])), max(0, int(xyxy[1])),
                          min(w, int(xyxy[2])), min(h, int(xyxy[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        raw_boxes.append((x1, y1, x2, y2))
        raw_confs.append(float(conf))
    boxes, confs = _filter_contained_boxes(raw_boxes, raw_confs)

    dets: list[Detection] = []
    for box, det_conf in zip(boxes, confs):
        crop = image.crop(box)
        x = tfm(crop).unsqueeze(0).to(device)

        mean_probs, uncertainty = mc_dropout_predict(classifier, x, n_passes=mc_passes)
        cls_conf, idx = mean_probs[0].max(0)
        class_name = class_names[int(idx)]

        logits, attn = classifier(x, return_attn=True)
        attn = attn[0].tolist()

        oci_score = None
        if oci_model is not None:
            reading = simulate_sensors(class_name, rng=rng)
            f_m = normalize_moisture(reading.moisture_raw, DEFAULT_ANCHORS)
            f_g = normalize_gas(reading.rs, DEFAULT_ANCHORS)
            oci_score = compute_oci(oci_model, f_m=f_m, f_g=f_g)

        decision = decide(class_name, float(cls_conf), float(uncertainty[0]), oci_score)

        dets.append(Detection(
            box=box, det_conf=det_conf, class_name=class_name, cls_conf=float(cls_conf),
            uncertainty=float(uncertainty[0]), attn_convnext=attn[0], attn_vit=attn[1],
            oci_score=oci_score, decision=decision,
        ))
    return dets


def _log_detection(logger: PredictionLogger | None, frame_idx: int, d: Detection):
    if logger is None:
        return
    logger.log(DetectionRecord(
        timestamp=dt.datetime.now().isoformat(timespec="seconds"),
        frame_idx=frame_idx, class_name=d.class_name, cls_conf=d.cls_conf,
        det_conf=d.det_conf, x1=d.box[0], y1=d.box[1], x2=d.box[2], y2=d.box[3],
        attn_convnext=d.attn_convnext, attn_vit=d.attn_vit,
        uncertainty=d.uncertainty, oci_score=d.oci_score if d.oci_score is not None else float("nan"),
        decision=d.decision.route if d.decision else "",
    ))


def run_images(cfg: Config, det_ckpt: str, cls_ckpt: str, inputs: list[str],
                log_dir: str | None = None):
    device = pick_device()
    classifier, class_names, tfm = load_for_inference(cls_ckpt, cfg, device)
    detector = load_detector(det_ckpt, device)
    conf = cfg.detect.conf_threshold
    oci_model = _fit_demo_oci_model()
    logger = PredictionLogger(Path(log_dir) / "detections.csv") if log_dir else None

    any_found = False
    for frame_idx, path in enumerate(_iter_paths(inputs)):
        any_found = True
        image = Image.open(path).convert("RGB")
        dets = run_frame(detector, classifier, tfm, list(class_names), image,
                          device, conf_threshold=conf, oci_model=oci_model)
        if not dets:
            print(f"{path.name:<40} -> no items detected (conf >= {conf})")
            continue
        for i, d in enumerate(dets):
            oci_str = f"oci={d.oci_score:.2f}" if d.oci_score is not None else "oci=n/a"
            print(f"{path.name:<40} item {i}: {d.class_name:<12} "
                  f"cls={d.cls_conf*100:5.1f}%  unc={d.uncertainty:.2f}  {oci_str}  "
                  f"-> {d.decision.route}")
            actuate_conveyor(d.decision)
            _log_detection(logger, frame_idx, d)
    if not any_found:
        print("No images found in the given inputs.")
    if logger:
        logger.close()


def run_camera(cfg: Config, det_ckpt: str, cls_ckpt: str, cam_index: int,
                log_dir: str | None = None):
    import cv2

    device = pick_device()
    classifier, class_names, tfm = load_for_inference(cls_ckpt, cfg, device)
    detector = load_detector(det_ckpt, device)
    conf = cfg.detect.conf_threshold
    oci_model = _fit_demo_oci_model()
    logger = PredictionLogger(Path(log_dir) / "detections.csv") if log_dir else None

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {cam_index}.")
    print("Detect+classify+decide demo running. 'q' quits, 'e' explains last frame.")
    last_dets: list[Detection] = []
    last_image: Image.Image | None = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            dets = run_frame(detector, classifier, tfm, list(class_names), image,
                              device, conf_threshold=conf, oci_model=oci_model, mc_passes=6)
            last_dets, last_image = dets, image
            for d in dets:
                x1, y1, x2, y2 = d.box
                color = (0, 255, 0) if d.decision.route == d.class_name else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{d.class_name} {d.cls_conf*100:.0f}% u={d.uncertainty:.2f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                actuate_conveyor(d.decision)
                _log_detection(logger, frame_idx, d)
            cv2.imshow("edge-waste (full pipeline)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e") and last_dets and last_image is not None and log_dir:
                _save_explanations(classifier, tfm, last_image, last_dets, log_dir)
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if logger:
            logger.close()


def _save_explanations(classifier, tfm, image: Image.Image, dets: list[Detection], log_dir: str):
    from .explain import explain_crop

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, d in enumerate(dets):
        crop = image.crop(d.box)
        x = tfm(crop).unsqueeze(0)
        x = x.to(next(classifier.parameters()).device)
        class_idx = None  # GradCAM without an explicit target explains the top prediction
        out = Path(log_dir) / f"explain_{ts}_item{i}.png"
        try:
            explain_crop(classifier, x, class_idx, out, original_crop=crop)
            print(f"class={d.class_name} conf={d.cls_conf:.2f} "
                  f"attn(convnext,vit)=({d.attn_convnext:.2f},{d.attn_vit:.2f}) -> {out}")
        except ImportError:
            print("pytorch-grad-cam not installed - `pip install grad-cam` to enable 'e'.")
            return


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Full detect-classify-decide pipeline demo.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--det-ckpt", required=True,
                     help="YOLO26 detector weights, e.g. runs/detect/taco_single_class/weights/best.pt")
    ap.add_argument("--cls-ckpt", default="runs/stage1/best.pt")
    ap.add_argument("--camera", action="store_true", help="Live webcam demo.")
    ap.add_argument("--cam-index", type=int, default=0)
    ap.add_argument("--log-dir", default=None,
                     help="Session directory for detections.csv (+ Grad-CAM PNGs in camera mode).")
    ap.add_argument("inputs", nargs="*", help="Image files or directories.")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    if args.camera:
        run_camera(cfg, args.det_ckpt, args.cls_ckpt, args.cam_index, log_dir=args.log_dir)
    elif args.inputs:
        run_images(cfg, args.det_ckpt, args.cls_ckpt, args.inputs, log_dir=args.log_dir)
    else:
        ap.error("Provide image paths/dirs, or use --camera.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
