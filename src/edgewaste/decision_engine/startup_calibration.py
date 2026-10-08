# startup_calibration.py - sets the uncertainty threshold and the conformal family thresholds from saved validation outputs when a program starts
"""Used by the live and video sorters. Everything is calibrated on the VALIDATION split of the verified first model
(runs/analysis/convnext_vit/cache_val.npz), never on the test split.
  * tau_u : the uncertainty value above which the least certain `review_budget` share of validation items falls
  * q     : one conformal threshold per material family (a stricter level for the hazardous family)"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from edgewaste.decision_engine.conformal_routing import alpha_vector, calibrate, family_matrix

ROOT = Path(__file__).resolve().parents[3]
CACHE_VAL = ROOT / "runs" / "analysis" / "convnext_vit" / "cache_val.npz"


@torch.no_grad()
def mc_head_uncertainty(head: torch.nn.Module, emb: torch.Tensor, passes: int = 25):
    """Monte Carlo dropout on the classifier head only (the backbones have no dropout, so this equals repeating the whole network).
    returns (mean probabilities [n, classes], normalised entropy [n] between 0 and 1)."""
    head.train()
    probs = torch.stack([torch.softmax(head(emb), 1) for _ in range(passes)]).mean(0)
    head.eval()
    entropy = -(probs * (probs + 1e-12).log()).sum(1) / np.log(probs.shape[1])
    return probs, entropy


def calibrate_from_cache(class_names, head, device, alpha: float = 0.10, alpha_h: float = 0.05, review_budget: float = 0.10,
                         cache_path: Path = CACHE_VAL):
    """returns (q per family, tau_u, family matrix M)"""
    d = np.load(cache_path, allow_pickle=True)
    if list(d["class_names"]) != list(class_names):
        raise ValueError("the saved validation outputs and the checkpoint use a different class order")
    M = family_matrix(list(class_names))
    probs = torch.softmax(torch.from_numpy(d["logits"].astype(np.float32)), 1).numpy().astype(np.float64)
    q = calibrate(probs @ M, M[d["labels"].astype(int)].argmax(1), alpha_vector(alpha, alpha_h))
    torch.manual_seed(0)
    _, u = mc_head_uncertainty(head, torch.from_numpy(d["emb"].astype(np.float32)).to(device))
    tau_u = float(np.quantile(u.cpu().numpy(), 1 - review_budget))
    return q, tau_u, M
