"""Two-stage inference: YOLO26 localizes items, the Stage 1 classifier types
each crop.

This is the tech-stack doc's "detect-then-classify" pipeline (section 2.3):
swapping/upgrading either stage independently is why detection (YOLO26,
ultralytics) and classification (ConvNeXt+ViT hybrid, timm/torch) stay in
separate modules that only meet here.

Usage:
    edgewaste-pipeline --det-ckpt runs/detect/taco_single_class/weights/best.pt \
        --cls-ckpt runs/stage1/best.pt path/to/image_or_dir
    edgewaste-pipeline --det-ckpt ... --cls-ckpt ... --camera
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .config import Config
from .data.dataset import build_transforms
from .infer import IMAGE_EXTS, _iter_paths, load_for_inference
from .utils import pick_device


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords
    det_conf: float
    class_name: str
    cls_conf: float


def load_detector(det_ckpt: str, device):
    from ultralytics import YOLO
    model = YOLO(det_ckpt)
    return model


@torch.no_grad()
def run_frame(
    detector, classifier, tfm, class_names: list[str], image: Image.Image,
    device, conf_threshold: float = 0.35, topk: int = 1,
) -> list[Detection]:
    """Detect items in a PIL image, then classify each crop."""
    results = detector.predict(image, conf=conf_threshold, verbose=False)
    dets: list[Detection] = []
    if not results:
        return dets
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return dets
    w, h = image.size
    for xyxy, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
        x1, y1, x2, y2 = (max(0, int(xyxy[0])), max(0, int(xyxy[1])),
                          min(w, int(xyxy[2])), min(h, int(xyxy[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))
        x = tfm(crop).unsqueeze(0).to(device)
        probs = F.softmax(classifier(x), dim=1)[0]
        cls_conf, idx = probs.max(0)
        dets.append(Detection(
            box=(x1, y1, x2, y2), det_conf=float(conf),
            class_name=class_names[int(idx)], cls_conf=float(cls_conf),
        ))
    return dets


def run_images(cfg: Config, det_ckpt: str, cls_ckpt: str, inputs: list[str]):
    device = pick_device()
    classifier, class_names, tfm = load_for_inference(cls_ckpt, cfg, device)
    detector = load_detector(det_ckpt, device)
    conf = cfg.detect.conf_threshold

    any_found = False
    for path in _iter_paths(inputs):
        any_found = True
        image = Image.open(path).convert("RGB")
        dets = run_frame(detector, classifier, tfm, list(class_names), image,
                          device, conf_threshold=conf)
        if not dets:
            print(f"{path.name:<40} -> no items detected (conf >= {conf})")
            continue
        for i, d in enumerate(dets):
            print(f"{path.name:<40} item {i}: {d.class_name:<12} "
                  f"cls={d.cls_conf*100:5.1f}%  det={d.det_conf*100:5.1f}%  "
                  f"box={d.box}")
    if not any_found:
        print("No images found in the given inputs.")


def run_camera(cfg: Config, det_ckpt: str, cls_ckpt: str, cam_index: int):
    import cv2

    device = pick_device()
    classifier, class_names, tfm = load_for_inference(cls_ckpt, cfg, device)
    detector = load_detector(det_ckpt, device)
    conf = cfg.detect.conf_threshold

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {cam_index}.")
    print("Detect+classify demo running. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            dets = run_frame(detector, classifier, tfm, list(class_names),
                              image, device, conf_threshold=conf)
            for d in dets:
                x1, y1, x2, y2 = d.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{d.class_name} {d.cls_conf*100:.0f}%"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                            cv2.LINE_AA)
            cv2.imshow("edge-waste (detect+classify)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
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
    ap.add_argument("inputs", nargs="*", help="Image files or directories.")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    if args.camera:
        run_camera(cfg, args.det_ckpt, args.cls_ckpt, args.cam_index)
    elif args.inputs:
        run_images(cfg, args.det_ckpt, args.cls_ckpt, args.inputs)
    else:
        ap.error("Provide image paths/dirs, or use --camera.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
