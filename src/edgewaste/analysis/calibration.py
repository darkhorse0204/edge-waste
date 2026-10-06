# calibration.py - checks whether predicted confidence matches real accuracy (ece, reliability diagram, temperature scaling)
"""Calibration: when the model says "90% sure", is it right 90% of the time?

Reports expected calibration error (ECE, 15 equal-width confidence bins),
maximum calibration error, negative log-likelihood and multi-class Brier score,
before and after temperature scaling. The single temperature T is fitted on
the validation split by minimising NLL of softmax(logits / T) and then applied
unchanged to test — it rescales confidence without changing any prediction
(argmax is invariant to dividing logits by a positive constant), so accuracy
is untouched and only the trustworthiness of the probabilities changes.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def calibration_stats(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece, mce = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()),
                     "accuracy": float(correct[m].mean()), "confidence": float(conf[m].mean())})
    onehot = np.eye(probs.shape[1])[y]
    return {
        "ece": float(ece), "mce": float(mce),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "mean_confidence": float(conf.mean()), "accuracy": float(correct.mean()),
        "bins": bins,
    }


def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    def nll(log_t):
        p = softmax(logits_val, np.exp(log_t))
        return -np.log(np.clip(p[np.arange(len(y_val)), y_val], 1e-12, 1)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def reliability_plot(before: dict, after: dict, T: float, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, s, title in ((axes[0], before, "Before (temperature = 1)"), (axes[1], after, f"After temperature scaling (temperature = {T:.2f})")):
        centers = [(b["lo"] + b["hi"]) / 2 for b in s["bins"]]
        ax.bar(centers, [b["accuracy"] for b in s["bins"]], width=1 / 15, edgecolor="black",
               color="#4c78a8", alpha=0.85, label="accuracy in bin")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
        ax.plot(centers, [b["confidence"] for b in s["bins"]], "o", color="#e45756", ms=4, label="mean confidence")
        ax.set_title(f"{title}\nExpected calibration error {s['ece'] * 100:.2f}%, negative log-likelihood {s['nll']:.3f}", fontsize=8)
        ax.set_xlabel("confidence"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy"); axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
