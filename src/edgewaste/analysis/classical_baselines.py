# classical_baselines.py - pca, lda, svm, knn, logistic regression, random forest and naive bayes baselines on three feature sets
"""Classical machine-learning baselines, to show what each stage of the deep
pipeline actually buys over classical methods.

Three feature sets, each standardised then reduced with PCA (keeping the
components that explain 95% of variance, capped at 256):
  handcrafted        HSV colour histogram + HOG            (classical CV)
  imagenet_frozen    ConvNeXt-Tiny ImageNet features       (transfer, no fine-tuning)
  finetuned_embedding  our hybrid's fused 1024-d embedding (after fine-tuning)

Classifiers: Gaussian naive Bayes, LDA (Ledoit-Wolf shrinkage), multinomial
logistic regression, linear SVM, RBF-kernel SVM, k-nearest neighbours and a
random forest. Hyperparameters are chosen on the validation split; the test
split is scored once with the chosen setting — the same protocol as the deep
model, so the numbers are directly comparable.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from edgewaste.analysis.metrics_report import family_and_hazard

FEATURE_SETS = {"handcrafted": "handcrafted", "imagenet_frozen": "imagenet",
                "finetuned_embedding": "emb"}

MODELS = {
    "gaussian_naive_bayes": (lambda: GaussianNB(), [{}]),
    "lda": (lambda **kw: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), [{}]),
    "logistic_regression": (lambda C: LogisticRegression(C=C, max_iter=3000), [{"C": c} for c in (0.01, 0.1, 1.0)]),
    "linear_svm": (lambda C: LinearSVC(C=C, max_iter=5000), [{"C": c} for c in (0.001, 0.01, 0.1)]),
    "rbf_svm": (lambda C: SVC(C=C, kernel="rbf", gamma="scale"), [{"C": c} for c in (1.0, 10.0)]),
    "knn": (lambda k: KNeighborsClassifier(n_neighbors=k, weights="distance"), [{"k": k} for k in (1, 5, 15)]),
    "random_forest": (lambda: RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=0), [{}]),
}


def pca_features(train: dict, val: dict, test: dict, key: str, var: float = 0.95, cap: int = 256):
    scaler = StandardScaler().fit(train[key].astype(np.float32))
    Xtr, Xva, Xte = (scaler.transform(d[key].astype(np.float32)) for d in (train, val, test))
    full = PCA(random_state=0).fit(Xtr)
    cum = np.cumsum(full.explained_variance_ratio_)
    n_at = {f"n_components_{int(v * 100)}pct": int(np.searchsorted(cum, v) + 1) for v in (0.90, 0.95, 0.99)}
    k = min(n_at[f"n_components_{int(var * 100)}pct"], cap)
    pca = PCA(n_components=k, random_state=0).fit(Xtr)
    info = {"raw_dim": int(Xtr.shape[1]), "kept_components": k,
            "variance_kept": float(pca.explained_variance_ratio_.sum()), **n_at,
            "cumulative_variance": cum[: min(len(cum), 512)].round(5).tolist()}
    return pca.transform(Xtr), pca.transform(Xva), pca.transform(Xte), info


def run_baselines(train: dict, val: dict, test: dict, class_names: list[str], log=print) -> dict:
    ytr, yva, yte = train["labels"].astype(int), val["labels"].astype(int), test["labels"].astype(int)
    results, pca_info, best_pred = [], {}, {}
    for fs_name, key in FEATURE_SETS.items():
        Xtr, Xva, Xte, info = pca_features(train, val, test, key)
        pca_info[fs_name] = info
        log(f"[{fs_name}] {info['raw_dim']}-d -> PCA {info['kept_components']} comps "
            f"({info['variance_kept'] * 100:.1f}% variance)")
        for m_name, (factory, grid) in MODELS.items():
            t0, best = time.time(), None
            for params in grid:
                clf = factory(**params).fit(Xtr, ytr)
                va = float((clf.predict(Xva) == yva).mean())
                if best is None or va > best[1]:
                    best = (params, va, clf)
            params, va, clf = best
            pte = clf.predict(Xte)
            row = {"feature_set": fs_name, "model": m_name, "params": params,
                   "train_accuracy": float((clf.predict(Xtr) == ytr).mean()),
                   "val_accuracy": va, "test_accuracy": float((pte == yte).mean()),
                   "test_macro_f1": float(f1_score(yte, pte, average="macro", zero_division=0)),
                   **{f"test_{k}": v for k, v in family_and_hazard(yte, pte, class_names).items()
                      if k in ("family_accuracy", "hazard_recall")},
                   "seconds": round(time.time() - t0, 1)}
            results.append(row)
            if fs_name not in best_pred or row["val_accuracy"] > best_pred[fs_name][0]:
                best_pred[fs_name] = (row["val_accuracy"], m_name, pte)
            log(f"  {m_name:22s} {str(params):14s} train {row['train_accuracy']:.3f} "
                f"val {va:.3f} test {row['test_accuracy']:.3f} ({row['seconds']}s)")
    return {"results": results, "pca": pca_info,
            "best_per_feature_set": {k: {"model": v[1], "test_predictions": v[2]} for k, v in best_pred.items()}}


def embedding_projections(test: dict, class_names: list[str], path, seed: int = 0) -> None:
    """2-D views of the test set: PCA and LDA of the fine-tuned embedding, and
    t-SNE of frozen ImageNet features vs fine-tuned embeddings — the visual
    answer to "what did fine-tuning change?"."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    from edgewaste.taxonomy import CLASS_TO_FAMILY, FAMILIES

    y = test["labels"].astype(int)
    fam_idx = np.array([FAMILIES.index(CLASS_TO_FAMILY[class_names[i]]) for i in y])
    emb = StandardScaler().fit_transform(test["emb"].astype(np.float32))
    img = StandardScaler().fit_transform(test["imagenet"].astype(np.float32))
    views = {
        "PCA (fine-tuned embedding)": PCA(2, random_state=seed).fit_transform(emb),
        "LDA (fine-tuned embedding)": LinearDiscriminantAnalysis(n_components=2).fit_transform(emb, y),
        "t-SNE (frozen ImageNet features)": TSNE(2, random_state=seed, init="pca").fit_transform(
            PCA(50, random_state=seed).fit_transform(img)),
        "t-SNE (fine-tuned embedding)": TSNE(2, random_state=seed, init="pca").fit_transform(
            PCA(50, random_state=seed).fit_transform(emb)),
    }
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (title, Z) in zip(axes.ravel(), views.items()):
        for f, fam in enumerate(FAMILIES):
            m = fam_idx == f
            ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.6, color=cmap(f), label=fam)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(markerscale=4, fontsize=8, loc="best")
    fig.suptitle("Test set (4,232 images) coloured by material family", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def pca_variance_plot(pca_info: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, info in pca_info.items():
        cum = info["cumulative_variance"]
        ax.plot(np.arange(1, len(cum) + 1), cum,
                label=f"{name} ({info['raw_dim']}-d, 95% at {info['n_components_95pct']} comps)")
    ax.axhline(0.95, ls="--", color="gray", lw=1)
    ax.set_xlabel("number of principal components"); ax.set_ylabel("cumulative explained variance")
    ax.set_xscale("log"); ax.set_ylim(0, 1.01); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("PCA explained variance per feature set")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
