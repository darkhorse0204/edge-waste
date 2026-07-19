"""Inference demo — the Stage 1 exit-criteria artifact.

Two modes:
  * image(s): classify one or more image files/dirs and print top-k predictions.
  * camera:   live webcam classification loop (OpenCV), overlaying the top
              prediction + confidence; this is the "basic inference demo running
              on the chosen edge device using the camera only" from docs/stage-1.

Confidence here is the plain softmax probability; the calibrated Bayesian /
MC-dropout confidence is a Stage 2 concern (Module 8).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .config import Config
from .data.dataset import build_transforms
from .models import build_model
from .utils import pick_device

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_for_inference(ckpt_path: str, cfg: Config, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = build_model(cfg.model, num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tfm = build_transforms(cfg.model.image_size, train=False)
    return model, class_names, tfm


@torch.no_grad()
def predict_image(model, tfm, class_names, path: Path, device, topk: int = 3):
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0]
    k = min(topk, len(class_names))
    conf, idx = probs.topk(k)
    return [(class_names[i], float(c)) for c, i in zip(conf, idx)]


def _iter_paths(inputs: list[str]):
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in IMAGE_EXTS:
                    yield f
        elif p.suffix.lower() in IMAGE_EXTS:
            yield p


def run_images(cfg: Config, ckpt: str, inputs: list[str], topk: int):
    device = pick_device()
    model, class_names, tfm = load_for_inference(ckpt, cfg, device)
    any_found = False
    for path in _iter_paths(inputs):
        any_found = True
        preds = predict_image(model, tfm, class_names, path, device, topk)
        top = preds[0]
        rest = "  ".join(f"{n}:{c:.2f}" for n, c in preds[1:])
        print(f"{path.name:<40} -> {top[0]:<14} {top[1]*100:5.1f}%   [{rest}]")
    if not any_found:
        print("No images found in the given inputs.")


def run_camera(cfg: Config, ckpt: str, cam_index: int, topk: int):
    import cv2

    device = pick_device()
    model, class_names, tfm = load_for_inference(ckpt, cfg, device)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {cam_index}.")
    print("Camera demo running. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)[0]
            conf, idx = probs.topk(min(topk, len(class_names)))
            label = f"{class_names[idx[0]]}  {conf[0]*100:.1f}%"
            cv2.putText(frame, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("edge-waste (Stage 1)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1 inference demo.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--ckpt", default="runs/stage1/best.pt")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--camera", action="store_true", help="Live webcam demo.")
    ap.add_argument("--cam-index", type=int, default=0)
    ap.add_argument("inputs", nargs="*", help="Image files or directories.")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    if args.camera:
        run_camera(cfg, args.ckpt, args.cam_index, args.topk)
    elif args.inputs:
        run_images(cfg, args.ckpt, args.inputs, args.topk)
    else:
        ap.error("Provide image paths/dirs, or use --camera.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
