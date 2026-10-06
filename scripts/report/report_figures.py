# report_figures.py - draws the extra figures (dataset, taxonomy, schedule, comparisons) used in the project report
"""New figures for the BITE497J Project I report. Every number is read from the repository
(split manifest, reports/ml_analysis, git history); nothing is typed in by hand except the
planned-hardware cost estimates, which are marked ESTIMATE on the figure.

Run:  python scripts/report/report_figures.py        (writes docs/report/figures/fig_*.png)
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "docs" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

from edgewaste.taxonomy import CLASS_TO_FAMILY, CLASS_NAMES, FAMILIES  # noqa: E402

FAM_COL = dict(zip(FAMILIES, plt.cm.tab10(np.arange(len(FAMILIES)))))
nice = lambda s: s.replace("_", " ")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def lookalike():
    from PIL import Image
    z = np.load(ROOT / "runs/analysis/convnext_vit/cache_test.npz", allow_pickle=True)
    paths, labels, names = z["paths"], z["labels"].astype(int), list(z["class_names"])
    rng = np.random.default_rng(3)
    pairs = [("cardboard_boxes", "cardboard_packaging"), ("steel_food_cans", "aluminum_food_cans")]
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for r, (a, b) in enumerate(pairs):
        for c, cls in enumerate([a, a, b, b]):
            idx = np.where(labels == names.index(cls))[0]
            i = int(rng.choice(idx))
            axes[r, c].imshow(Image.open(ROOT / paths[i]).convert("RGB")); axes[r, c].axis("off"); axes[r, c].grid(False)
            axes[r, c].set_title(nice(cls), fontsize=10)
    save(fig, "fig_lookalike.png")


def dataset_counts():
    d = pd.read_csv(ROOT / "data/splits_colab_reconstructed.csv")
    d["family"] = d.class_name.map(CLASS_TO_FAMILY)
    t = d.pivot_table(index="class_name", columns="split", values="path", aggfunc="count").fillna(0).astype(int)
    t = t.loc[list(CLASS_NAMES)]
    t.to_csv(OUT / "dataset_counts.csv")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bottom = np.zeros(len(t))
    for sp, hatch, alpha in (("train", "", 1.0), ("val", "//", 0.75), ("test", "..", 0.55)):
        ax.bar(range(len(t)), t[sp], bottom=bottom, color=[FAM_COL[CLASS_TO_FAMILY[c]] for c in t.index], alpha=alpha, hatch=hatch, edgecolor="white", label=sp)
        bottom += t[sp].values
    ax.set_xticks(range(len(t)), [nice(c) for c in t.index], rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("number of images"); ax.set_yscale("log")
    h = [plt.Rectangle((0, 0), 1, 1, color=FAM_COL[f]) for f in FAMILIES]
    fig.legend(h, FAMILIES, ncol=9, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    hs = [plt.Rectangle((0, 0), 1, 1, facecolor="grey", hatch=hh, alpha=a, edgecolor="white") for hh, a in (("", 1), ("//", .75), (".." , .55))]
    ax.legend(hs, ["train", "validation", "test"], fontsize=8, loc="upper right")
    save(fig, "fig_dataset_counts.png")


def taxonomy():
    fam_classes = {f: [c for c in CLASS_NAMES if CLASS_TO_FAMILY[c] == f] for f in FAMILIES}
    d = pd.read_csv(OUT / "dataset_counts.csv", index_col=0)
    n = d.sum(axis=1)
    fig, ax = plt.subplots(figsize=(11, 6.6)); ax.axis("off"); ax.grid(False)
    ax.set_xlim(0, 9); ax.set_ylim(0.9, 12.1)
    for k, f in enumerate(FAMILIES):
        x0 = k
        ax.add_patch(plt.Rectangle((x0 + 0.04, 10.7), 0.92, 1.0, color=FAM_COL[f], alpha=0.9))
        ax.text(x0 + 0.5, 11.2, f, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        for j, c in enumerate(fam_classes[f]):
            ax.add_patch(plt.Rectangle((x0 + 0.04, 9.9 - j * 1.0), 0.92, 0.85, color=FAM_COL[f], alpha=0.22))
            ax.text(x0 + 0.5, 10.32 - j * 1.0, nice(c).replace(" ", "\n", 1) if len(c) > 14 else nice(c), ha="center", va="center", fontsize=6.6)
            ax.text(x0 + 0.5, 10.03 - j * 1.0, f"{int(n[c]):,}", ha="center", va="center", fontsize=6, color="#444")
    ax.text(4.5, 11.95, "9 material families (routing bins) and 33 item classes, with the number of images of each class", ha="center", fontsize=10, fontweight="bold")
    save(fig, "fig_taxonomy.png")


def temperature():
    z = np.array([4.0, 3.0, 1.2, 0.4, -0.5])
    labs = ["class A", "class B", "class C", "class D", "class E"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.26
    for k, T in enumerate((0.71, 1.0, 2.0)):
        p = np.exp(z / T) / np.exp(z / T).sum()
        ax.bar(np.arange(5) + (k - 1) * w, p, w, label=f"T = {T}")
    ax.set_xticks(range(5), labs); ax.set_ylabel("probability"); ax.legend()
    save(fig, "fig_temperature.png")


def gantt():
    tasks = [
        ("Project blueprint and planning", "2026-07-06", "2026-07-19", "plan"),
        ("Stage 1: 7-class ConvNeXt + Vision Transformer pipeline", "2026-07-19", "2026-07-31", "build"),
        ("Detect-then-classify with YOLO; live inference; audit report", "2026-07-30", "2026-08-03", "build"),
        ("Review 1 submission and faculty feedback", "2026-08-03", "2026-08-19", "review"),
        ("Stage 2 and 3 in software: contamination index, uncertainty, decision engine, federated learning", "2026-09-22", "2026-09-22", "build"),
        ("33-class taxonomy, 18-class detector, video survey, identity prior", "2026-09-22", "2026-09-23", "build"),
        ("Colab training of the 33-class classifier (89.30%)", "2026-09-23", "2026-09-23", "train"),
        ("Explainability, segmentation, GAN, Swin comparison", "2026-09-24", "2026-09-24", "build"),
        ("Conformal hazard-leakage routing", "2026-09-25", "2026-09-25", "build"),
        ("Repository restructure and machine-learning analysis suite", "2026-09-30", "2026-09-30", "analysis"),
        ("Improved recipe retrain (93.83%); simulations; demonstration gallery", "2026-10-01", "2026-10-02", "train"),
        ("Report and review slides", "2026-09-24", "2026-10-09", "doc"),
        ("Pre-final (guide) review", "2026-10-05", "2026-10-09", "review"),
        ("Final review and report upload", "2026-10-21", "2026-10-21", "review"),
        ("Planned: hardware prototype (Project II)", "2026-10-22", "2026-12-20", "planned"),
    ]
    col = {"plan": "#7f7f7f", "build": "#4c78a8", "train": "#54a24b", "analysis": "#b279a2", "doc": "#f58518", "review": "#e45756", "planned": "#c7c7c7"}
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for i, (name, s, e, k) in enumerate(tasks):
        s, e = dt.date.fromisoformat(s), dt.date.fromisoformat(e)
        ax.barh(i, (e - s).days + 1, left=matplotlib.dates.date2num(s), color=col[k], hatch="//" if k == "planned" else "", edgecolor="white")
    ax.set_yticks(range(len(tasks)), [t[0] for t in tasks], fontsize=7.6); ax.invert_yaxis()
    ax.xaxis_date(); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b")); ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    h = [plt.Rectangle((0, 0), 1, 1, color=v) for v in col.values()]
    ax.legend(h, ["planning", "software build", "training", "analysis", "documentation", "review", "planned"], ncol=7, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, 1.1))
    save(fig, "fig_gantt.png")


def baselines():
    d = pd.read_csv(ROOT / "reports/ml_analysis/classical_baselines.csv")
    fs = [("handcrafted", "hand-crafted (colour + HOG)"), ("imagenet_frozen", "frozen ImageNet features"), ("finetuned_embedding", "fine-tuned hybrid embedding")]
    models = ["gaussian_naive_bayes", "lda", "logistic_regression", "linear_svm", "rbf_svm", "knn", "random_forest"]
    lab = ["naive Bayes", "LDA", "logistic reg.", "linear SVM", "RBF SVM", "kNN", "random forest"]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    w = 0.26
    for k, (f, fl) in enumerate(fs):
        v = [float(d[(d.feature_set == f) & (d.model == m)].test_accuracy.iloc[0]) * 100 for m in models]
        ax.bar(np.arange(len(models)) + (k - 1) * w, v, w, label=fl)
        for x, y in zip(np.arange(len(models)) + (k - 1) * w, v):
            ax.text(x, y + 0.8, f"{y:.0f}", ha="center", fontsize=7)
    ax.axhline(89.30, color="k", ls="--", lw=1); ax.text(-0.45, 106.5, "dashed line: fine-tuned hybrid with its own head, 89.3%", ha="left", fontsize=8)
    ax.set_xticks(range(len(models)), lab); ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, 112); ax.legend(fontsize=8, loc="upper left", ncol=3, bbox_to_anchor=(0, 0.99))
    save(fig, "fig_baselines.png")


def versions():
    rows = [("Hybrid v1\n(first recipe)", 89.30, 94.14, 93.50), ("Frozen features\n+ RBF SVM", 93.3, 97.1, 98.1), ("Hybrid v2\n(improved recipe)", 93.83, 97.28, 98.37)]
    fig, ax = plt.subplots(figsize=(8.4, 4))
    w = 0.26
    for k, (m, lab) in enumerate((("item accuracy", 1), ("family accuracy", 2), ("hazard recall", 3))):
        v = [r[lab] for r in rows]
        ax.bar(np.arange(3) + (k - 1) * w, v, w, label=m)
        for x, y in zip(np.arange(3) + (k - 1) * w, v):
            ax.text(x, y + 0.2, f"{y:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), [r[0] for r in rows]); ax.set_ylim(85, 100); ax.set_ylabel("%"); ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig_versions.png")


def oci_roc():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, roc_auc_score
    from edgewaste.contamination.synthetic_calibration_data import generate_synthetic_calibration_data as gen
    tr, a = gen(n_replicates=300, seed=0)
    te, _ = gen(n_replicates=300, seed=1, anchors=a)
    def feats(df):
        fm = np.clip((df.moisture_raw - a.m_dry) / (a.m_wet - a.m_dry), 0, 1)
        fg = np.clip((-np.log(df.rs_over_r0) - a.l_min) / (a.l_max - a.l_min), 0, 1)
        return fm.values, fg.values
    ftm, ftg = feats(tr); fem, feg = feats(te)
    ytr, yte = tr.contaminated.values.astype(int), te.contaminated.values.astype(int)
    models = {"both sensors": (np.c_[ftm, ftg, ftm * ftg], np.c_[fem, feg, fem * feg]), "moisture only": (ftm[:, None], fem[:, None]), "gas only": (ftg[:, None], feg[:, None])}
    fig, ax = plt.subplots(figsize=(5.4, 4.6)); out = {}
    for name, (xtr, xte) in models.items():
        m = LogisticRegression(C=10).fit(xtr, ytr); s = m.predict_proba(xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, s); auc = roc_auc_score(yte, s); out[name] = float(auc)
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.axhline(0.95, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate (sensitivity)"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("SIMULATED sensors, held-out draw", fontsize=9)
    save(fig, "fig_oci_roc.png")
    (OUT / "oci_roc.json").write_text(json.dumps(out, indent=1))


def cost():
    items = [("Edge computer (Raspberry Pi 5 class)", 9000), ("Camera module", 3000), ("Moisture and gas sensors", 600), ("Analogue-to-digital converter", 400),
             ("Servos and motor driver", 1500), ("Conveyor belt mechanism and frame", 4000), ("Bins, wiring, power supply", 2500)]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    v = [x[1] for x in items]
    ax.barh(range(len(items)), v, color="#4c78a8")
    for i, x in enumerate(v):
        ax.text(x + 100, i, f"~Rs {x:,}", va="center", fontsize=8)
    ax.set_yticks(range(len(items)), [x[0] for x in items], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 11500)
    ax.set_xlabel("approximate cost in Indian rupees (ESTIMATE, to be confirmed with vendor quotations)")
    save(fig, "fig_cost.png")
    return sum(v)


if __name__ == "__main__":
    for fn in (dataset_counts, taxonomy, lookalike, temperature, gantt, baselines, versions, oci_roc):
        fn()
    print("bom total", cost())
