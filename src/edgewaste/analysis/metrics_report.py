# metrics_report.py - full classification metric suite with bootstrap confidence intervals and mcnemar model comparison
"""Every metric a reviewer is likely to ask for, computed from cached
predictions: accuracy, balanced accuracy, macro/weighted precision-recall-F1,
Cohen's kappa, Matthews correlation, top-k accuracy, log loss, one-vs-rest
ROC-AUC and PR-AUC, family-level (routing) accuracy and hazard recall — with
95% bootstrap confidence intervals for the headline numbers, so a difference
between two models can be judged against sampling noise.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score,
                             top_k_accuracy_score)
from sklearn.preprocessing import label_binarize

from edgewaste.taxonomy import CLASS_TO_FAMILY, is_hazardous


def family_and_hazard(y: np.ndarray, pred: np.ndarray, class_names: list[str]) -> dict:
    fam = np.array([CLASS_TO_FAMILY[n] for n in class_names])
    haz = np.array([is_hazardous(n) for n in class_names])
    true_h, pred_h = haz[y], haz[pred]
    return {
        "family_accuracy": float((fam[y] == fam[pred]).mean()),
        "hazard_recall": float(pred_h[true_h].mean()),
        "hazard_precision": float(true_h[pred_h].mean()) if pred_h.any() else 0.0,
        "hazard_caught": int(pred_h[true_h].sum()),
        "hazard_support": int(true_h.sum()),
    }


def full_metrics(y: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    pred = probs.argmax(1)
    labels = np.arange(len(class_names))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y, pred, labels=labels, average="weighted", zero_division=0)
    y_bin = label_binarize(y, classes=labels)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(p_m), "macro_recall": float(r_m), "macro_f1": float(f_m),
        "weighted_precision": float(p_w), "weighted_recall": float(r_w), "weighted_f1": float(f_w),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y, pred)),
        "top3_accuracy": float(top_k_accuracy_score(y, probs, k=3, labels=labels)),
        "top5_accuracy": float(top_k_accuracy_score(y, probs, k=5, labels=labels)),
        "log_loss": float(log_loss(y, np.clip(probs, 1e-12, 1), labels=labels)),
        "macro_roc_auc_ovr": float(roc_auc_score(y_bin, probs, average="macro")),
        "macro_pr_auc": float(average_precision_score(y_bin, probs, average="macro")),
        **family_and_hazard(y, pred, class_names),
    }


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, class_names: list[str],
                 n_boot: int = 1000, seed: int = 0) -> dict:
    """Percentile 95% intervals by resampling test items with replacement."""
    rng = np.random.default_rng(seed)
    fam = np.array([CLASS_TO_FAMILY[n] for n in class_names])
    haz = np.array([is_hazardous(n) for n in class_names])
    stats = {"accuracy": [], "macro_f1": [], "family_accuracy": [], "hazard_recall": []}
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yi, pi = y[i], pred[i]
        stats["accuracy"].append((yi == pi).mean())
        stats["macro_f1"].append(f1_score(yi, pi, average="macro", labels=np.arange(len(class_names)), zero_division=0))
        stats["family_accuracy"].append((fam[yi] == fam[pi]).mean())
        stats["hazard_recall"].append(haz[pi][haz[yi]].mean())
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}


def per_class_table(y: np.ndarray, pred: np.ndarray, class_names: list[str]) -> list[dict]:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=np.arange(len(class_names)), zero_division=0)
    return [{"class": n, "family": CLASS_TO_FAMILY[n], "precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f[i]), "support": int(s[i]), "predicted": int((pred == i).sum())}
            for i, n in enumerate(class_names)]


def top_confusions(y: np.ndarray, pred: np.ndarray, class_names: list[str], k: int = 10) -> list[dict]:
    pairs: dict[tuple[int, int], int] = {}
    for t, p in zip(y[y != pred], pred[y != pred]):
        pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:k]
    return [{"true": class_names[t], "predicted": class_names[p], "count": c,
             "same_family": CLASS_TO_FAMILY[class_names[t]] == CLASS_TO_FAMILY[class_names[p]]}
            for (t, p), c in top]


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Paired test on the same test items: do models A and B differ in error
    rate beyond chance? Exact binomial for small discordant counts,
    continuity-corrected chi-square otherwise."""
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int((a_ok & ~b_ok).sum())  # a right, b wrong
    c = int((~a_ok & b_ok).sum())  # a wrong, b right
    if b + c < 25:
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        return {"a_right_b_wrong": b, "a_wrong_b_right": c, "test": "exact binomial", "p_value": float(p)}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "test": "chi2 (continuity corrected)",
            "statistic": float(stat), "p_value": float(chi2.sf(stat, 1))}
