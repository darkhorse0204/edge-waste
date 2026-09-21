"""MC-Dropout Bayesian uncertainty estimation (blueprint Module 8).

Plain softmax confidence is overconfident on out-of-distribution crops (a
hand, a shadow, an unfamiliar material). MC-Dropout instead runs N
stochastic forward passes with dropout left active at inference time and
reports the *spread* across passes as an uncertainty score — no extra
training, no new data, works on the classifier that's already trained.

Uncertainty here is predictive entropy of the mean softmax distribution
across passes, normalised to [0, 1] by dividing by log(num_classes) (the
entropy of a uniform distribution, i.e. maximum possible uncertainty).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _enable_dropout(model: torch.nn.Module) -> None:
    """Set only Dropout layers to train mode; everything else (BatchNorm
    etc.) stays in eval mode so running statistics aren't disturbed."""
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module, x: torch.Tensor, n_passes: int = 25
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run N stochastic forward passes.

    Returns (mean_probs [B, C], uncertainty [B]) where uncertainty is
    normalised predictive entropy in [0, 1] — 0 confident, 1 maximally
    uncertain (uniform over classes).
    """
    was_training = model.training
    model.eval()
    _enable_dropout(model)

    probs_stack = []
    for _ in range(n_passes):
        logits = model(x)
        probs_stack.append(F.softmax(logits, dim=1))
    probs = torch.stack(probs_stack, dim=0)  # [N, B, C]
    mean_probs = probs.mean(dim=0)  # [B, C]

    num_classes = mean_probs.shape[-1]
    eps = 1e-12
    entropy = -(mean_probs * (mean_probs + eps).log()).sum(dim=-1)  # [B]
    uncertainty = entropy / math.log(num_classes)

    model.train(was_training)
    return mean_probs, uncertainty
