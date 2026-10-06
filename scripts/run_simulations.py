# run_simulations.py - simulations that test each claimed mechanism on real held-out data (vision) or synthetic sensors (labelled)
"""Evidence runs for the four mechanisms in the invention disclosure.

Real data, real model (the ConvNeXt+ViT checkpoint runs/stage1 and its recovered,
verified test split; nothing here ever touches training images):

  uncertainty   MC-dropout uncertainty: does it flag errors? risk-coverage curve
  conformal     hazard leakage vs review workload, and validity of the bound
  ablation      routing outcomes with each mechanism added in turn
  shift         the same mechanisms under image corruption (noise, blur, darkness,
                occlusion) - where the conformal guarantee's exchangeability
                assumption is violated and the uncertainty gate must take over

Synthetic sensors (no physical hardware exists; every output is labelled synthetic):

  oci           dedicated single-channel model vs imputing a default value, when
                one sensor channel drops out, across sensor-noise levels

Prerequisite: python scripts/collect_analysis_features.py  (caches classifier outputs)
Usage:        python scripts/run_simulations.py [--only uncertainty conformal ablation shift oci history]
Writes:       reports/simulations/summary.json and reports/simulations/figures/*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edgewaste.analysis.collect_predictions import load_split  # noqa: E402
from edgewaste.decision_engine.conformal_routing import (HAZ, alpha_vector, calibrate,  # noqa: E402
                                                         family_matrix, prediction_sets)
from edgewaste.taxonomy import FAMILIES  # noqa: E402

OUT = Path("reports/simulations")
FIG = OUT / "figures"
CACHE = Path("runs/analysis/convnext_vit/cache_test.npz")
CKPT = Path("runs/stage1/best.pt")
MANIFEST = "data/splits_colab_reconstructed.csv"
TAU_U = 0.5          # normalised-entropy review threshold used by the decision engine
N_MC = 25            # stochastic passes
NL = chr(10)
REVIEW_BUDGET = 0.10  # calibrated gates: the 10% least-certain calibration items define the threshold
COLORS = {"top1": "#7f7f7f", "msp": "#9ecae9", "mc": "#4c78a8", "conf": "#f58518", "conf_mc": "#54a24b", "conf_msp": "#b279a2"}
LABELS = {"top1": "top-1 class only", "msp": "maximum-probability gate", "mc": "Monte Carlo dropout gate",
          "conf": "conformal family sets", "conf_mc": "conformal sets + Monte Carlo dropout gate",
          "conf_msp": "conformal sets + maximum-probability gate"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    return plt


def load_summary() -> dict:
    p = OUT / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_summary(S: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float))


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------
def load_head() -> tuple[nn.Sequential, float]:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    dropout = ck["model_cfg"]["dropout"]
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(1024, 512), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(512, ck["model_state"]["head.4.weight"].shape[0]))
    head.load_state_dict({k[5:]: v for k, v in ck["model_state"].items() if k.startswith("head.")})
    return head.eval(), dropout


@torch.no_grad()
def mc_uncertainty(head: nn.Sequential, emb: torch.Tensor, passes: int = N_MC, seed: int = 0):
    """Normalised predictive entropy of the mean of `passes` dropout passes. The
    backbones contain no active dropout (drop rate 0), so repeating only the head
    is mathematically the same as repeating the whole network, ~100x cheaper."""
    torch.manual_seed(seed)
    head.train()
    probs = torch.stack([torch.softmax(head(emb), 1) for _ in range(passes)]).mean(0)
    head.eval()
    ent = -(probs * (probs + 1e-12).log()).sum(1)
    return probs, ent / np.log(probs.shape[1])


def load_clean():
    d = load_split(CACHE)
    names = d["class_names"]
    M = family_matrix(names)
    y = d["labels"].astype(int)
    probs = d["probs"].astype(np.float64)
    return {"names": names, "M": M, "y": y, "fam_true": M[y].argmax(1), "probs": probs,
            "fam_mass": probs @ M, "fam_top1": M[probs.argmax(1)].argmax(1),
            "emb": torch.from_numpy(d["emb"].astype(np.float32))}


def routing(kind: str, fam_top1, fam_mass, U, q, conf=None, tau_u=TAU_U, tau_c=None) -> tuple[np.ndarray, np.ndarray]:
    """action: 0 review, 1 automatic non-hazard bin, 2 automatic hazard bin; bin: family index."""
    n = len(fam_top1)
    action, bin_ = np.zeros(n, int), np.full(n, -1)
    if kind in ("top1", "mc", "msp"):
        ok = {"top1": np.ones(n, bool), "mc": None if U is None else U < tau_u,
              "msp": None if conf is None else conf >= tau_c}[kind]
        action[ok] = np.where(fam_top1[ok] == HAZ, 2, 1)
        bin_[ok] = fam_top1[ok]
    else:
        sets = prediction_sets(fam_mass, q)
        single = sets.sum(1) == 1
        fam = np.where(single, sets.argmax(1), -1)
        ok = single if kind == "conf" else (single & (U < tau_u) if kind == "conf_mc" else single & (conf >= tau_c))
        action[ok] = np.where(fam[ok] == HAZ, 2, 1)
        bin_[ok] = fam[ok]
    return action, bin_


def outcomes(action, bin_, fam_true) -> dict:
    haz = fam_true == HAZ
    auto = action > 0
    correct = auto & (bin_ == fam_true)
    n = len(fam_true)
    return {
        "correct": float(correct.sum() / n),
        "hazard_leak_share": float((haz & (action == 1)).sum() / n),
        "false_hazard_share": float((~haz & (action == 2)).sum() / n),
        "other_misroute_share": float((~haz & (action == 1) & ~correct).sum() / n),
        "review": float((~auto).sum() / n),
        # rates used in the text
        "hazard_leak": float((haz & (action == 1)).sum() / max(haz.sum(), 1)),
        "false_hazard_rate": float((~haz & (action == 2)).sum() / max((~haz).sum(), 1)),
        "auto_rate": float(auto.mean()),
        "accuracy_of_auto": float(correct.sum() / max(auto.sum(), 1)),
    }


# ---------------------------------------------------------------------------
# a. uncertainty
# ---------------------------------------------------------------------------
def sim_uncertainty(C, S):
    from sklearn.metrics import roc_auc_score
    head, _ = load_head()
    with torch.no_grad():
        agree = (head(C["emb"]).argmax(1).numpy() == C["probs"].argmax(1)).mean()
    mc_p, U = mc_uncertainty(head, C["emb"])
    U = U.numpy()
    pred = C["probs"].argmax(1)
    wrong = (pred != C["y"]).astype(int)
    fam_wrong = (C["fam_top1"] != C["fam_true"]).astype(int)
    conf = C["probs"].max(1)
    auroc = {"item_errors_by_U": roc_auc_score(wrong, U), "item_errors_by_1_minus_maxsoftmax": roc_auc_score(wrong, 1 - conf),
             "routing_errors_by_U": roc_auc_score(fam_wrong, U), "routing_errors_by_1_minus_maxsoftmax": roc_auc_score(fam_wrong, 1 - conf)}
    order_u, order_c = np.argsort(U), np.argsort(-conf)
    cov = np.linspace(0.05, 1.0, 96)
    risk = lambda order, kind: [1 - (kind[order[: max(1, int(c * len(order)))]]).mean() for c in cov]
    ok_item, ok_fam = 1 - wrong, 1 - fam_wrong
    S["uncertainty"] = {
        "head_only_matches_full_network_argmax": float(agree),
        "mean_U_correct": float(U[wrong == 0].mean()), "mean_U_wrong": float(U[wrong == 1].mean()),
        "fraction_U_ge_tau": float((U >= TAU_U).mean()), "tau_u": TAU_U, "max_U": float(U.max()),
        "auroc": {k: float(v) for k, v in auroc.items()},
        "accuracy_at_coverage": {f"{int(c * 100)}%": {"by_U": float(ok_item[order_u[: int(c * len(U))]].mean()),
                                                       "by_maxsoftmax": float(ok_item[order_c[: int(c * len(U))]].mean())}
                                 for c in (0.5, 0.7, 0.9, 1.0)},
        "family_accuracy_at_coverage": {f"{int(c * 100)}%": {"by_U": float(ok_fam[order_u[: int(c * len(U))]].mean()),
                                                              "by_maxsoftmax": float(ok_fam[order_c[: int(c * len(U))]].mean())}
                                        for c in (0.5, 0.7, 0.9, 1.0)},
    }
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.linspace(0, max(0.6, U.max()), 40)
    ax[0].hist(U[wrong == 0], bins, alpha=0.7, label=f"correct (n={int((wrong == 0).sum())})", color="#54a24b", density=True)
    ax[0].hist(U[wrong == 1], bins, alpha=0.7, label=f"wrong (n={int(wrong.sum())})", color="#e45756", density=True)
    ax[0].set_xlabel("Monte Carlo dropout uncertainty (normalised entropy)"); ax[0].set_ylabel("density"); ax[0].legend()
    ax[0].set_title("(a) Uncertainty of right and wrong predictions")
    for kind, order, lab, c in ((ok_item, order_u, "ranked by Monte Carlo dropout uncertainty", "#4c78a8"), (ok_item, order_c, "ranked by maximum class probability", "#f58518")):
        ax[1].plot(cov * 100, [100 * (1 - r) for r in risk(order, kind)], label=lab, color=c)
    ax[1].set_xlabel("items accepted automatically (%)"); ax[1].set_ylabel("item accuracy of accepted items (%)")
    ax[1].legend(); ax[1].set_title("(b) Accuracy of accepted items")
    for kind, order, lab, c in ((ok_fam, order_u, "ranked by Monte Carlo dropout uncertainty", "#4c78a8"), (ok_fam, order_c, "ranked by maximum class probability", "#f58518")):
        ax[2].plot(cov * 100, [100 * (1 - r) for r in risk(order, kind)], label=lab, color=c)
    ax[2].set_xlabel("items accepted automatically (%)"); ax[2].set_ylabel("routing (family) accuracy of accepted items (%)")
    ax[2].legend(); ax[2].set_title("(c) Same, at the routing level")
    fig.tight_layout(); fig.savefig(FIG / "fig_uncertainty.png", dpi=130); plt.close(fig)
    return U


# ---------------------------------------------------------------------------
# c. conformal operating curve + validity of the bound
# ---------------------------------------------------------------------------
def sim_conformal(C, S, trials: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(C["y"])
    conf = C["probs"].max(1)
    alpha_hs = [0.20, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005]
    rows = []
    for ah in alpha_hs:
        alphas = alpha_vector(0.10, ah)
        r_c, r_b = [], []
        for _ in range(trials):
            perm = rng.permutation(n); cal, ev = perm[: n // 2], perm[n // 2:]
            q = calibrate(C["fam_mass"][cal], C["fam_true"][cal], alphas)
            a, b = routing("conf", C["fam_top1"][ev], C["fam_mass"][ev], None, q)
            r_c.append(outcomes(a, b, C["fam_true"][ev]))
            ac, bc = routing("conf", C["fam_top1"][cal], C["fam_mass"][cal], None, q)
            tau = float(np.quantile(conf[cal], float((ac == 0).mean())))
            ok = conf[ev] >= tau
            a2 = np.zeros(len(ev), int); a2[ok] = np.where(C["fam_top1"][ev][ok] == HAZ, 2, 1)
            b2 = np.where(ok, C["fam_top1"][ev], -1)
            r_b.append(outcomes(a2, b2, C["fam_true"][ev]))
        agg = lambda rs, k: (float(np.mean([r[k] for r in rs])), float(np.percentile([r[k] for r in rs], 95)))
        rows.append({"alpha_hazard": ah,
                     "conformal": {"leak_mean": agg(r_c, "hazard_leak")[0], "leak_p95": agg(r_c, "hazard_leak")[1],
                                   "review": float(np.mean([r["review"] for r in r_c])),
                                   "misroute": float(np.mean([r["other_misroute_share"] + r["hazard_leak_share"] + r["false_hazard_share"] for r in r_c]))},
                     "conf_gate": {"leak_mean": agg(r_b, "hazard_leak")[0], "leak_p95": agg(r_b, "hazard_leak")[1],
                                   "review": float(np.mean([r["review"] for r in r_b]))}})
    a0, b0 = routing("top1", C["fam_top1"], None, None, None)
    top1 = outcomes(a0, b0, C["fam_true"])
    S["conformal_curve"] = {"alpha_nonhazard": 0.10, "trials": trials, "top1_hazard_leak": top1["hazard_leak"], "rows": rows}
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    rs_c = sorted(rows, key=lambda r: r["conformal"]["review"]); rs_b = sorted(rows, key=lambda r: r["conf_gate"]["review"])
    ax[0].plot([r["conformal"]["review"] * 100 for r in rs_c], [r["conformal"]["leak_mean"] * 100 for r in rs_c], "o-", color="#f58518", label="conformal family sets (mean)")
    ax[0].plot([r["conformal"]["review"] * 100 for r in rs_c], [r["conformal"]["leak_p95"] * 100 for r in rs_c], "o:", color="#f58518", alpha=0.7, label="conformal (95th percentile)")
    ax[0].plot([r["conf_gate"]["review"] * 100 for r in rs_b], [r["conf_gate"]["leak_mean"] * 100 for r in rs_b], "s-", color="#4c78a8", label="maximum-probability gate tuned to the same workload")
    ax[0].axhline(top1["hazard_leak"] * 100, color="#7f7f7f", ls="--", label=f"top-1 routing, no gate ({top1['hazard_leak'] * 100:.1f}%)")
    for r in rows:
        if r["alpha_hazard"] in (0.05, 0.02, 0.01, 0.005):
            ax[0].annotate(f"α_H={r['alpha_hazard']}", (r["conformal"]["review"] * 100, r["conformal"]["leak_mean"] * 100), fontsize=8, xytext=(6, -12), textcoords="offset points")
    ax[0].set_xlabel("items sent to human review (%)"); ax[0].set_ylabel("true hazards auto-routed to a recycling bin (%)")
    ax[0].set_title("(a) Hazard leakage for the same human workload"); ax[0].legend(fontsize=8)
    ah = np.array([r["alpha_hazard"] for r in rows])
    ax[1].plot(ah * 100, ah * 100, "k--", lw=1, label="target bound α_H")
    ax[1].plot(ah * 100, [r["conformal"]["leak_mean"] * 100 for r in rows], "o-", color="#f58518", label="measured leakage (mean)")
    ax[1].plot(ah * 100, [r["conformal"]["leak_p95"] * 100 for r in rows], "o:", color="#f58518", label="measured (95th percentile)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("operator-set hazard error level α_H (%)"); ax[1].set_ylabel("measured hazard leakage (%)")
    ax[1].set_title("(b) The measured leakage stays under the set bound"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_conformal_curve.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# e. routing outcomes with each mechanism added (clean data, calibration/evaluation halves)
# ---------------------------------------------------------------------------
def sim_ablation(C, S, U, trials: int = 200, seed: int = 1):
    rng = np.random.default_rng(seed)
    n = len(C["y"]); alphas = alpha_vector(0.10, 0.05)
    conf = C["probs"].max(1)
    acc = {k: [] for k in COLORS}; acc_default = []
    for _ in range(trials):
        perm = rng.permutation(n); cal, ev = perm[: n // 2], perm[n // 2:]
        q = calibrate(C["fam_mass"][cal], C["fam_true"][cal], alphas)
        tau_u = float(np.quantile(U[cal], 1 - REVIEW_BUDGET)); tau_c = float(np.quantile(conf[cal], REVIEW_BUDGET))
        for k in COLORS:
            a, b = routing(k, C["fam_top1"][ev], C["fam_mass"][ev], U[ev], q, conf[ev], tau_u, tau_c)
            acc[k].append(outcomes(a, b, C["fam_true"][ev]))
        a, b = routing("mc", C["fam_top1"][ev], C["fam_mass"][ev], U[ev], q, tau_u=TAU_U)   # the fixed 0.5 default
        acc_default.append(outcomes(a, b, C["fam_true"][ev]))
    S["ablation"] = {"alpha": 0.10, "alpha_hazard": 0.05, "review_budget_for_gates": REVIEW_BUDGET, "trials": trials,
                     **{k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in acc.items()},
                     "mc_default_tau_0.5": {m: float(np.mean([r[m] for r in acc_default])) for m in acc_default[0]}}
    plt = _plt()
    cats = [("correct", "correct bin", "#54a24b"), ("review", "sent to human review", "#9ecae9"),
            ("other_misroute_share", "wrong recycling bin", "#f58518"),
            ("false_hazard_share", "non-hazard sent to hazard bin", "#b279a2"),
            ("hazard_leak_share", "HAZARD sent to a recycling bin", "#e45756")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(COLORS)); bottom = np.zeros(len(COLORS))
    for key, lab, col in cats:
        vals = np.array([S["ablation"][k][key] * 100 for k in COLORS])
        ax.bar(x, vals, bottom=bottom, label=lab, color=col, width=0.62)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 2.2: ax.text(xi, b + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=8)
        bottom += vals
    ax.set_xticks(x, [LABELS[k].replace(" gate", NL + "gate").replace("conformal ", "conformal" + NL) for k in COLORS], fontsize=7.5)
    ax.set_ylabel("share of all items (%)"); ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    ax.set_title("(a) Where every item ends up (gates calibrated to a 10% review budget)", fontsize=9)
    leak = [S["ablation"][k]["hazard_leak"] * 100 for k in COLORS]; rev = [S["ablation"][k]["review"] * 100 for k in COLORS]
    ax2.barh(x, leak, color=[COLORS[k] for k in COLORS])
    for xi, l, r in zip(x, leak, rev):
        ax2.text(l + 0.1, xi, f"{l:.2f}%   (review {r:.1f}%)", va="center", fontsize=8)
    ax2.set_yticks(x, [LABELS[k] for k in COLORS], fontsize=8); ax2.invert_yaxis(); ax2.set_xlim(0, max(leak) * 1.8)
    ax2.set_xlabel("true hazards auto-routed to a recycling bin (%)")
    ax2.set_title("(b) Hazard leakage and the review workload it costs", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_routing_ablation.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# b. distribution shift
# ---------------------------------------------------------------------------
def corrupt(x01: torch.Tensor, kind: str, level: float, gen: torch.Generator) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    if kind == "noise":
        return (x01 + torch.randn(x01.shape, generator=gen) * level).clamp(0, 1)
    if kind == "blur":
        k = int(4 * level) * 2 + 1
        return TF.gaussian_blur(x01, [k, k], [level, level])
    if kind == "darkness":
        return x01 * level
    if kind == "occlusion":
        out = x01.clone(); side = int(224 * np.sqrt(level))
        for i in range(len(out)):
            r, c = (torch.randint(0, 224 - side + 1, (2,), generator=gen)).tolist()
            out[i, :, r:r + side, c:c + side] = 0.0
        return out
    raise ValueError(kind)


CONDITIONS = [("clean", None, 0.0)] + [(k, k, v) for k, vs in (
    ("noise", (0.05, 0.10, 0.20)), ("blur", (1.0, 2.0, 4.0)),
    ("darkness", (0.5, 0.3, 0.15)), ("occlusion", (0.10, 0.25, 0.50))) for v in vs]


def sim_shift(C, S, U_clean, n_eval: int = 1500, seed: int = 0):
    from torch.utils.data import DataLoader, Subset
    from edgewaste.classification.evaluate_classifier import _load_model
    from edgewaste.config import Config
    from edgewaste.data.waste_image_dataset import IMAGENET_MEAN, IMAGENET_STD, WasteDataset

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(C["y"])); ev_idx, cal_idx = perm[:n_eval], perm[n_eval:]
    q = calibrate(C["fam_mass"][cal_idx], C["fam_true"][cal_idx], alpha_vector(0.10, 0.05))
    conf_all = C["probs"].max(1)
    tau_u = float(np.quantile(U_clean[cal_idx], 1 - REVIEW_BUDGET)); tau_c = float(np.quantile(conf_all[cal_idx], REVIEW_BUDGET))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_model(CKPT, Config.load("configs/classifier_convnext_vit.yaml"), device)
    ds = WasteDataset(MANIFEST, "test", 224, train=False)
    xs = torch.cat([x[None] for x, _ in (ds[int(i)] for i in ev_idx)]).half()   # normalised, fp16 on cpu
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    y_ev, M = C["y"][ev_idx], C["M"]
    fam_true = C["fam_true"][ev_idx]
    results = []
    for name, kind, level in CONDITIONS:
        gen = torch.Generator().manual_seed(1234)
        embs = []
        with torch.no_grad():
            for i in range(0, n_eval, 50):
                xb = xs[i:i + 50].float()
                if kind:
                    xb = (corrupt(xb * std + mean, kind, level, gen) - mean) / std
                xb = xb.to(device)
                f = model.fusion([model.proj_cnx(model.convnext(xb)), model.proj_vit(model.vit(xb))])[0]
                embs.append(f.cpu())
        emb = torch.cat(embs)
        with torch.no_grad():
            model.head.eval(); logits = model.head(emb.to(device)).cpu()
        probs = torch.softmax(logits, 1).numpy().astype(np.float64)
        _, U = mc_uncertainty(model.head, emb.to(device))
        U = U.cpu().numpy()
        fm = probs @ M; ft1 = M[probs.argmax(1)].argmax(1)
        mx = probs.max(1)
        row = {"condition": name, "level": level, "item_accuracy": float((probs.argmax(1) == y_ev).mean()),
               "family_accuracy": float((ft1 == fam_true).mean()), "mean_U": float(U.mean()),
               "frac_U_above_tau": float((U >= tau_u).mean()), "mean_max_softmax": float(mx.mean()),
               "frac_msp_below_tau": float((mx < tau_c).mean())}
        for k in COLORS:
            a, b = routing(k, ft1, fm, U, q, mx, tau_u, tau_c)
            row[k] = outcomes(a, b, fam_true)
        results.append(row)
        print(f"  shift {name:9s} {level:5}: acc {row['item_accuracy']:.3f}  meanU {row['mean_U']:.3f}  "
              + "  ".join(f"{k}: leak {row[k]['hazard_leak']:.3f}/fa {row[k]['false_hazard_rate']:.3f}/auto {row[k]['auto_rate']:.2f}" for k in COLORS), flush=True)
    S["shift"] = {"n_eval": n_eval, "n_calibration": int(len(cal_idx)), "alpha": 0.10, "alpha_hazard": 0.05,
                  "tau_u_calibrated": tau_u, "tau_msp_calibrated": tau_c, "review_budget": REVIEW_BUDGET, "results": results}
    plt = _plt()
    kinds = ["noise", "blur", "darkness", "occlusion"]
    clean = results[0]
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), sharex="col")
    for j, kd in enumerate(kinds):
        rs = [clean] + [r for r in results if r["condition"] == kd]
        xs_ = [0] + [r["level"] for r in rs[1:]] if False else list(range(len(rs)))
        labels = ["clean"] + [str(r["level"]) for r in rs[1:]]
        axes[0, j].plot(xs_, [r["item_accuracy"] * 100 for r in rs], "o-", color="#333", label="item accuracy")
        axes[0, j].plot(xs_, [r["frac_U_above_tau"] * 100 for r in rs], "s--", color="#4c78a8", label="% flagged by Monte Carlo dropout gate")
        axes[0, j].plot(xs_, [r["frac_msp_below_tau"] * 100 for r in rs], "^--", color="#9ecae9", label="% flagged by maximum-probability gate")
        axes[0, j].set_title(kd); axes[0, j].set_ylabel("item accuracy / % flagged")
        for i, (key, ttl, yl) in enumerate((("hazard_leak", "true hazards auto-routed to a recycling bin", "%"),
                                            ("false_hazard_rate", "non-hazards auto-sent to the hazard bin", "%"),
                                            ("auto_rate", "items routed automatically", "%")), start=1):
            for k in COLORS:
                axes[i, j].plot(xs_, [r[k][key] * 100 for r in rs], "o-", color=COLORS[k], label=LABELS[k])
            axes[i, j].set_ylabel(ttl, fontsize=7)
        axes[3, j].set_xticks(xs_, labels); axes[3, j].set_xlabel("severity")
    axes[0, 0].legend(fontsize=7); axes[1, 0].legend(fontsize=7)
    fig.suptitle(f"Behaviour under image corruption ({n_eval} test images; conformal thresholds calibrated on clean images only)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(FIG / "fig_distribution_shift.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
# d. synthetic sensors: dropout of one channel
# ---------------------------------------------------------------------------
def sim_oci(S, n_rep: int = 300):
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from edgewaste.contamination import (compute_oci, fit_oci_weights, normalize_gas,
                                         normalize_moisture, select_threshold)
    from edgewaste.contamination.synthetic_calibration_data import generate_synthetic_calibration_data as gen

    def feats(df, an):
        return (df["moisture_raw"].apply(lambda v: normalize_moisture(v, an)).to_numpy(),
                df["rs_over_r0"].apply(lambda v: normalize_gas(v, an)).to_numpy(), df["contaminated"].to_numpy())

    sweep = []
    for scale in (0.5, 1.0, 2.0):
        tr, an = gen(n_replicates=n_rep, seed=1, noise_scale=scale)
        te, _ = gen(n_replicates=n_rep, seed=2, noise_scale=scale, anchors=an)
        fm_tr, fg_tr, y_tr = feats(tr, an); fm_te, fg_te, y_te = feats(te, an)
        model, _ = fit_oci_weights(fm_tr, fg_tr, y_tr)

        def score(model_, fm, fg, case, impute=None):
            # impute=none -> the dedicated single-channel model; a number -> combined model with that value
            if case == "both": return np.array([compute_oci(model_, a, b) for a, b in zip(fm, fg)])
            if case == "moisture_only": return np.array([compute_oci(model_, a, impute) for a in fm])   # gas lost
            return np.array([compute_oci(model_, impute, b) for b in fg])                               # moisture lost

        tr_full = score(model, fm_tr, fg_tr, "both"); te_full = score(model, fm_te, fg_te, "both")
        thr_full = select_threshold(y_tr, tr_full, 0.95)["operating_threshold"]
        full_dec = te_full >= thr_full
        entry = {"noise_scale": scale, "full": {"auc": float(roc_auc_score(y_te, te_full)),
                 "sens": float(full_dec[y_te == 1].mean()), "fpr": float(full_dec[y_te == 0].mean()),
                 "brier": float(brier_score_loss(y_te, te_full))}, "cases": {}}
        for case in ("moisture_only", "gas_only"):
            kept_tr, kept_te = (fm_tr, fm_te) if case == "moisture_only" else (fg_tr, fg_te)
            lost_tr = fg_tr if case == "moisture_only" else fm_tr
            methods = {}
            ded_tr = score(model, fm_tr, fg_tr, case); ded_te = score(model, fm_te, fg_te, case)
            thr_own = select_threshold(y_tr, ded_tr, 0.95)["operating_threshold"]
            methods["dedicated model (own calibrated threshold)"] = (ded_te, thr_own)
            for lab, fill in (("impute 0, shared threshold", 0.0), ("impute training mean, shared threshold", float(lost_tr.mean()))):
                methods[lab] = (score(model, fm_te, fg_te, case, impute=fill), thr_full)
            imp_tr = score(model, fm_tr, fg_tr, case, impute=0.0)
            methods["impute 0, threshold re-tuned for the dropout case"] = (score(model, fm_te, fg_te, case, impute=0.0),
                                                                          select_threshold(y_tr, imp_tr, 0.95)["operating_threshold"])
            cres = {}
            for lab, (s, thr) in methods.items():
                d = s >= thr
                cres[lab] = {"auc": float(roc_auc_score(y_te, s)), "sens": float(d[y_te == 1].mean()), "fpr": float(d[y_te == 0].mean()),
                             "brier": float(brier_score_loss(y_te, s)), "flip_rate_vs_full_sensors": float((d != full_dec).mean()),
                             "mean_score_contaminated": float(s[y_te == 1].mean()), "mean_score_clean": float(s[y_te == 0].mean())}
            entry["cases"][case] = cres
        sweep.append(entry)
        print(f"  oci noise x{scale}: full AUC {entry['full']['auc']:.3f} sens {entry['full']['sens']:.3f} fpr {entry['full']['fpr']:.3f}", flush=True)
        for case, cres in entry["cases"].items():
            for lab, r in cres.items():
                print(f"     {case:14s} {lab:52s} auc {r['auc']:.3f} sens {r['sens']:.3f} fpr {r['fpr']:.3f} brier {r['brier']:.3f} flips {r['flip_rate_vs_full_sensors']:.3f}", flush=True)
    S["oci_dropout_synthetic"] = {"n_train": 10 * n_rep, "n_test": 10 * n_rep, "target_sensitivity": 0.95, "sweep": sweep}

    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cols = {"dedicated model (own calibrated threshold)": "#54a24b", "impute 0, shared threshold": "#e45756",
            "impute training mean, shared threshold": "#f58518", "impute 0, threshold re-tuned for the dropout case": "#9ecae9"}
    for r_i, case in enumerate(("moisture_only", "gas_only")):
        for c_i, (metric, ttl) in enumerate((("sens", "sensitivity (target ≥ 0.95)"), ("fpr", "false-positive rate"), ("brier", "Brier score (lower = better calibrated)"))):
            ax = axes[r_i, c_i]; w = 0.2
            for m_i, lab in enumerate(cols):
                nice = lab.replace("impute training mean", "training average as default").replace("impute 0", "default value 0").replace("re-tuned", "tuned again")
                ax.bar(np.arange(3) + (m_i - 1.5) * w, [e["cases"][case][lab][metric] for e in sweep], w, label=nice, color=cols[lab])
            ax.set_xticks(range(3), [f"noise ×{e['noise_scale']}" for e in sweep]); ax.set_title(f"{case.replace('_', ' ')}: {ttl}", fontsize=9)
            if metric == "sens": ax.axhline(0.95, color="black", ls="--", lw=1)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("SIMULATED sensor data: one channel lost - dedicated single-channel model versus using a default value", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(FIG / "fig_oci_dropout_synthetic.png", dpi=125); plt.close(fig)


# ---------------------------------------------------------------------------
# training history of the improved recipe
# ---------------------------------------------------------------------------
def fig_history(S):
    h = json.loads(Path("runs/stage1_v2/history.json").read_text())
    ep = [r["epoch"] for r in h]
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [r["train_loss"] for r in h], "o-", label="train loss (augmented, label-smoothed)")
    ax[0].plot(ep, [r["val_loss"] for r in h], "s-", label="validation loss")
    ax[0].axvspan(0.5, 2.5, color="#9ecae9", alpha=0.3, label="backbones frozen (epochs 1-2)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(fontsize=8); ax[0].set_title("Loss")
    ax[1].plot(ep, [r["train_acc"] * 100 for r in h], "o-", label="train accuracy (augmented)")
    ax[1].plot(ep, [r["val_acc"] * 100 for r in h], "s-", label="validation accuracy")
    ax[1].axvspan(0.5, 2.5, color="#9ecae9", alpha=0.3)
    best = int(np.argmax([r["val_acc"] for r in h]))
    ax[1].annotate(f"best {h[best]['val_acc'] * 100:.2f}% (epoch {ep[best]})", (ep[best], h[best]["val_acc"] * 100), xytext=(-60, -40), textcoords="offset points", arrowprops={"arrowstyle": "->"})
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy (%)"); ax[1].legend(fontsize=8); ax[1].set_title("Accuracy")
    fig.suptitle("Improved training recipe (new layers first, then backbones at one tenth of the learning rate)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig_training_history_v2.png", dpi=130); plt.close(fig)
    S["training_history_v2"] = {"epochs": len(h), "best_val_acc": float(h[best]["val_acc"]), "best_epoch": ep[best]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", default=["uncertainty", "conformal", "ablation", "shift", "oci", "history"])
    args = ap.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    S = load_summary()
    need_clean = any(a in args.only for a in ("uncertainty", "conformal", "ablation", "shift"))
    if need_clean:
        C = load_clean()
        head, _ = load_head()
        _, U = mc_uncertainty(head, C["emb"]); U = U.numpy()
    for step in args.only:
        print(f"[{step}]", flush=True)
        if step == "uncertainty": sim_uncertainty(C, S)
        elif step == "conformal": sim_conformal(C, S)
        elif step == "ablation": sim_ablation(C, S, U)
        elif step == "shift": sim_shift(C, S, U)
        elif step == "oci": sim_oci(S)
        elif step == "history": fig_history(S)
        save_summary(S)
    print(f"wrote {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
