"""Export the trained classifier (and optionally the YOLO26 detector) to
ONNX for edge deployment, verifying numerical parity against PyTorch.

Usage:
    edgewaste-export --ckpt runs/stage1/best.pt --out runs/stage1/model.onnx
    edgewaste-export --ckpt runs/stage1/best.pt --out runs/stage1/model.onnx \
        --det-ckpt runs/detect/taco_single_class/weights/best.pt \
        --det-out runs/detect/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .infer import load_for_inference


def export_classifier(ckpt: str, cfg: Config, out_path: str, atol: float = 1e-3) -> Path:
    device = torch.device("cpu")  # export from CPU for a portable graph
    model, class_names, _ = load_for_inference(ckpt, cfg, device)
    model.eval()

    dummy = torch.randn(1, 3, cfg.model.image_size, cfg.model.image_size)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    _verify_parity(model, out_path, dummy, atol)
    print(f"Classifier exported: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB), "
          f"classes={list(class_names)}")
    return out_path


def _verify_parity(model: torch.nn.Module, onnx_path: Path, dummy: torch.Tensor, atol: float) -> None:
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    if max_diff > atol:
        raise RuntimeError(
            f"ONNX/PyTorch parity check FAILED: max abs diff {max_diff:.2e} > atol {atol:.2e}. "
            "Do not deploy this export."
        )
    print(f"Parity OK: max abs diff {max_diff:.2e} (atol {atol:.2e})")


def export_detector(det_ckpt: str, out_path: str, image_size: int = 640) -> Path:
    """One-line export via ultralytics' own exporter."""
    from ultralytics import YOLO

    model = YOLO(det_ckpt)
    exported = model.export(format="onnx", imgsz=image_size, simplify=True)
    out_path = Path(out_path)
    if str(exported) != str(out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Path(exported).replace(out_path)
    print(f"Detector exported: {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export models to ONNX with parity verification.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--ckpt", default="runs/stage1/best.pt")
    ap.add_argument("--out", default="runs/stage1/model.onnx")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--det-ckpt", default=None, help="Also export the YOLO26 detector.")
    ap.add_argument("--det-out", default="runs/detect/model.onnx")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)

    export_classifier(args.ckpt, cfg, args.out, atol=args.atol)
    if args.det_ckpt:
        export_detector(args.det_ckpt, args.det_out, image_size=cfg.detect.image_size)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
