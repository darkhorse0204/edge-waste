# model_complexity.py - parameter counts per component, flops, inference latency and file sizes
"""Model cost: how big and how fast. Parameter count per component, FLOPs for
one 224x224 image (counted by torch's own FLOP counter, 1 multiply-add = 2
FLOPs), batch-1 latency on GPU and CPU, and checkpoint / ONNX sizes."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode


def _params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def _latency_ms(model, device, runs: int, warmup: int) -> float:
    x = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def complexity(model: torch.nn.Module, ckpt_path: str, onnx_path: str | None = None) -> dict:
    model = model.eval().cpu()
    parts = {"convnext_backbone": model.convnext, "vit_backbone": model.vit,
             "projections": torch.nn.ModuleList([model.proj_cnx, model.proj_vit]),
             "attention_fusion": model.fusion, "classifier_head": model.head}
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    out = {"total_parameters": _params(model),
           "parameters_by_component": {k: _params(v) for k, v in parts.items()},
           "gflops_per_image": fc.get_total_flops() / 1e9,
           "cpu_latency_ms_batch1": _latency_ms(model, torch.device("cpu"), runs=10, warmup=2),
           "checkpoint_mb": Path(ckpt_path).stat().st_size / 1e6}
    if torch.cuda.is_available():
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["gpu_latency_ms_batch1"] = _latency_ms(model.cuda(), torch.device("cuda"), runs=50, warmup=10)
        model.cpu()
    if onnx_path and Path(onnx_path).exists():
        out["onnx_mb"] = Path(onnx_path).stat().st_size / 1e6
    return out
