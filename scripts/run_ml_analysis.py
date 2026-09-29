# run_ml_analysis.py - runs the full ml analysis suite on cached features and writes reports/ml_analysis
"""One command for every analysis a reviewer may ask about.

Prerequisite (once, ~10 min on the RTX 2050):
    python scripts/reconstruct_colab_split.py
    python scripts/collect_analysis_features.py --expect-acc 0.8930

Then:
    python scripts/run_ml_analysis.py            # everything, ~20-40 min (classical baselines dominate)
    python scripts/run_ml_analysis.py --skip-baselines --skip-complexity   # quick re-run

Writes reports/ml_analysis/summary.json, CSV tables and figures/*.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edgewaste.analysis import calibration as cal  # noqa: E402
from edgewaste.analysis import class_imbalance as imb  # noqa: E402
from edgewaste.analysis import classical_baselines as cb  # noqa: E402
from edgewaste.analysis import fit_diagnostics as fd  # noqa: E402
from edgewaste.analysis.collect_predictions import load_split  # noqa: E402
from edgewaste.analysis.metrics_report import (bootstrap_ci, full_metrics, mcnemar,  # noqa: E402
                                               per_class_table, top_confusions)
from edgewaste.classification.evaluate_classifier import _plot_confusion  # noqa: E402
from edgewaste.taxonomy import CLASS_TO_FAMILY, FAMILIES  # noqa: E402


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in rows)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def imbalance_figure(dist, share_before, share_after, sweep, class_names, tau, path):
    plt = _plt()
    counts = np.array([dist["train"]["counts"][n] for n in class_names])
    order = np.argsort(-counts)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].bar(range(len(order)), counts[order], color="#4c78a8")
    axes[0].set_xticks(range(len(order)), [class_names[i] for i in order], rotation=90, fontsize=7)
    axes[0].set_title(f"Training images per class (imbalance ratio {dist['train']['imbalance_ratio']:.1f}x)")
    x = np.arange(len(order))
    axes[1].bar(x - 0.2, share_before[order], 0.4, label="as trained", color="#e45756")
    axes[1].bar(x + 0.2, share_after[order], 0.4, label=f"logit-adjusted (tau={tau})", color="#54a24b")
    axes[1].axhline(1, color="black", lw=1)
    axes[1].set_xticks(x, [class_names[i] for i in order], rotation=90, fontsize=7)
    axes[1].set_title("Predicted / true count per class on test\n(>1 = over-predicted), frequent -> rare")
    axes[1].legend(fontsize=8)
    taus = [r["tau"] for r in sweep["val"]]
    for key, style in (("accuracy", "o-"), ("balanced_accuracy", "s-"), ("macro_f1", "^-"), ("hazard_recall", "d--")):
        axes[2].plot(taus, [r[key] for r in sweep["test"]], style, label=f"test {key}")
    axes[2].axvline(tau, color="gray", ls=":")
    axes[2].set_xlabel("tau (logit adjustment strength)"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    axes[2].set_title("Post-hoc logit adjustment sweep (tau chosen on val)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def per_class_figure(rows, path):
    plt = _plt()
    rows = sorted(rows, key=lambda r: r["f1"])
    fig, ax = plt.subplots(figsize=(8, 9))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r["precision"] for r in rows], 0.4, label="precision", color="#4c78a8")
    ax.barh(y + 0.2, [r["recall"] for r in rows], 0.4, label="recall", color="#f58518")
    ax.set_yticks(y, [f"{r['class']} (n={r['support']})" for r in rows], fontsize=7)
    ax.set_xlim(0, 1); ax.legend(); ax.grid(alpha=0.3, axis="x")
    ax.set_title("Per-class precision / recall on test (sorted by F1)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def attention_figure(att: dict, path):
    plt = _plt()
    fams = list(att)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(fams, [att[f]["convnext"] for f in fams], label="ConvNeXt", color="#4c78a8")
    ax.bar(fams, [att[f]["vit"] for f in fams], bottom=[att[f]["convnext"] for f in fams], label="ViT", color="#f58518")
    ax.axhline(0.5, color="black", lw=1, ls="--")
    ax.set_ylabel("mean fusion attention weight"); ax.legend(); plt.xticks(rotation=30)
    ax.set_title("How much the fused classifier relies on each backbone (test set)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", default="runs/analysis/convnext_vit")
    ap.add_argument("--ckpt", default="runs/stage1/best.pt")
    ap.add_argument("--config", default="configs/classifier_convnext_vit.yaml")
    ap.add_argument("--onnx", default="runs/stage1/model.onnx")
    ap.add_argument("--history", default="runs/stage1/history.json",
                    help="per-epoch history of the analysed run (download it from Drive if trained on Colab)")
    ap.add_argument("--fallback-history", default="runs/stage1_swin/history.json")
    ap.add_argument("--out", default="reports/ml_analysis")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-complexity", action="store_true")
    args = ap.parse_args(argv)

    out, figs = Path(args.out), Path(args.out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    S = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    cache = Path(args.cache_dir)
    tr, va, te = (load_split(cache / f"cache_{s}.npz") for s in ("train", "val", "test"))
    names = te["class_names"]

    def save():  # after every section, so a late failure never loses earlier results
        summary_path.write_text(json.dumps(S, indent=1, default=float))
    y_tr, y_va, y_te = (d["labels"].astype(int) for d in (tr, va, te))
    pred_te = te["probs"].argmax(1)

    print("1/7 metrics")
    S["metrics"] = {"train_eval_mode": full_metrics(y_tr, tr["probs"], names),
                    "val": full_metrics(y_va, va["probs"], names),
                    "test": full_metrics(y_te, te["probs"], names)}
    S["bootstrap_95ci_test"] = bootstrap_ci(y_te, pred_te, names)
    S["top_confusions_test"] = top_confusions(y_te, pred_te, names, k=12)
    pc = per_class_table(y_te, pred_te, names)
    _write_csv(pc, out / "per_class_metrics_test.csv")
    per_class_figure(pc, figs / "per_class_precision_recall.png")
    _plot_confusion(y_te, pred_te, list(range(len(names))), names, figs / "confusion_matrix_items.png")
    fam = np.array([FAMILIES.index(CLASS_TO_FAMILY[n]) for n in names])
    _plot_confusion(fam[y_te], fam[pred_te], list(range(len(FAMILIES))), list(FAMILIES),
                    figs / "confusion_matrix_families.png")
    save()

    print("2/7 fit diagnostics")
    gap = fd.generalisation_gap({"train": tr, "val": va, "test": te})
    Xtr, Xva, _, _ = cb.pca_features(tr, va, te, "imagenet")
    lc = fd.learning_curve(Xtr, y_tr, Xva, y_va)
    vc = fd.validation_curve(Xtr, y_tr, Xva, y_va)
    history = fd.load_history(args.history)
    hist_label = "main model"
    if history is None:
        history = fd.load_history(args.fallback_history)
        hist_label = "ConvNeXt+Swin run - main run's history.json is on Drive"
    S["fit_diagnostics"] = {"generalisation_gap": gap, "checkpoint": fd.checkpoint_metadata(args.ckpt),
                            "per_class_gap_top10": fd.per_class_gap(tr, te, names)[:10],
                            "learning_curve": lc, "validation_curve": vc,
                            "history_source": hist_label if history else None, "history": history}
    fd.plot_fit(gap, lc, vc, history, hist_label, figs / "fit_diagnostics.png")
    save()

    print("3/7 calibration")
    T = cal.fit_temperature(va["logits"], y_va)
    before = cal.calibration_stats(cal.softmax(te["logits"]), y_te)
    after = cal.calibration_stats(cal.softmax(te["logits"], T), y_te)
    S["calibration_test"] = {"temperature_fitted_on_val": T,
                             "before": {k: v for k, v in before.items() if k != "bins"},
                             "after": {k: v for k, v in after.items() if k != "bins"}}
    cal.reliability_plot(before, after, T, figs / "reliability_diagram.png")
    save()

    print("4/7 class imbalance")
    dist = imb.class_distribution({"train": y_tr, "val": y_va, "test": y_te}, names)
    train_counts = np.bincount(y_tr, minlength=len(names))
    sweep = imb.tau_sweep(va["logits"], y_va, te["logits"], y_te, train_counts, names)
    tau = sweep["best_tau_on_val"]
    pred_adj = imb.adjusted_predictions(te["logits"], train_counts, tau)
    share_b = imb.prediction_share(y_te, pred_te, len(names))
    share_a = imb.prediction_share(y_te, pred_adj, len(names))
    S["class_imbalance"] = {"distribution": dist, "logit_adjustment": sweep,
                            "test_prediction_share_as_trained": dict(zip(names, share_b.round(3).tolist())),
                            "test_prediction_share_adjusted": dict(zip(names, share_a.round(3).tolist())),
                            "corr_log_train_count_vs_log_share": float(np.corrcoef(
                                np.log(train_counts), np.log(np.maximum(share_b, 1e-3)))[0, 1])}
    imbalance_figure(dist, share_b, share_a, sweep, names, tau, figs / "class_imbalance.png")
    save()

    print("5/7 backbone attention")
    att = {}
    for f in FAMILIES:
        m = np.array([CLASS_TO_FAMILY[names[i]] == f for i in y_te])
        w = te["attn"][m].astype(np.float32).mean(0)
        att[f] = {"convnext": float(w[0]), "vit": float(w[1])}
    S["fusion_attention_by_family"] = att
    attention_figure(att, figs / "fusion_attention.png")
    save()

    if not args.skip_baselines:
        print("6/7 classical baselines (PCA / LDA / SVM / kNN / LR / RF / NB)")
        base = cb.run_baselines(tr, va, te, names)
        _write_csv(base["results"], out / "classical_baselines.csv")
        S["classical_baselines"] = {"results": base["results"],
                                    "pca": {k: {kk: vv for kk, vv in v.items() if kk != "cumulative_variance"}
                                            for k, v in base["pca"].items()},
                                    "mcnemar_deep_vs_best_classical": {
                                        fs: {"classical_model": v["model"], **mcnemar(y_te, pred_te, v["test_predictions"])}
                                        for fs, v in base["best_per_feature_set"].items()}}
        cb.pca_variance_plot(base["pca"], figs / "pca_explained_variance.png")
        cb.embedding_projections(te, names, figs / "embedding_projections.png")
        save()

    if not args.skip_complexity:
        print("7/7 model complexity")
        from edgewaste.analysis.model_complexity import complexity
        from edgewaste.classification.evaluate_classifier import _load_model
        from edgewaste.config import Config
        import torch
        model, _ = _load_model(Path(args.ckpt), Config.load(args.config), torch.device("cpu"))
        S["complexity"] = complexity(model, args.ckpt, args.onnx)

    save()
    print(f"wrote {summary_path} and {len(list(figs.glob('*.png')))} figures in {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
