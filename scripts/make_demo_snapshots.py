# make_demo_snapshots.py - builds the demonstration figures and worked-example numbers used in the invention disclosure draft
"""Demonstration snapshots for the invention disclosure draft.

Outputs (all from real held-out images and the stored classifier outputs):
  reports/demo/routing_demo.png        routing decisions on eight held-out images
  reports/demo/explain_panel.png       original | Grad-CAM | LIME | sensor contribution bars
  reports/demo/video/video_frames.png  annotated frames of the video-survey run
  reports/demo/worked_example.json     the real numbers quoted in the equation explanations

Run:  python scripts/make_demo_snapshots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_simulations as rs  # noqa: E402
from edgewaste.decision_engine.conformal_routing import (HAZ, alpha_vector, calibrate,  # noqa: E402
                                                         prediction_sets)
from edgewaste.taxonomy import FAMILIES  # noqa: E402

OUT = Path("reports/demo")
SEED = 7


def _short(name: str) -> str:
    return name.replace("_", " ")


def routing_demo() -> dict:
    from PIL import Image
    plt = rs._plt()
    C = rs.load_clean()
    raw = np.load(rs.CACHE, allow_pickle=True)
    paths = raw["paths"]
    head, _ = rs.load_head()
    probs_mc, U = rs.mc_uncertainty(head, C["emb"])
    U = U.numpy()
    n = len(C["y"])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    cal, ev = perm[: n // 2], perm[n // 2:]
    alphas = alpha_vector(0.10, 0.05)
    q = calibrate(C["fam_mass"][cal], C["fam_true"][cal], alphas)
    tau_u = float(np.quantile(U[cal], 1 - rs.REVIEW_BUDGET))
    sets = prediction_sets(C["fam_mass"], q)

    top1 = C["probs"].argmax(1)
    haz_true = C["fam_true"] == HAZ
    size = sets.sum(1)
    is_ev = np.zeros(n, bool); is_ev[ev] = True
    single = size == 1

    def pick(mask, k=1):
        idx = np.where(mask & is_ev)[0]
        return list(rng.choice(idx, size=min(k, len(idx)), replace=False))

    wanted = []
    labels = []
    for mask, tag, k in [
        (single & ~haz_true & (U < tau_u) & (C["fam_top1"] == C["fam_true"]) & (C["fam_true"] != HAZ), "automatic, correct bin", 3),
        (single & haz_true & (sets[:, HAZ]) & (U < tau_u), "hazard to hazardous bin", 1),
        (haz_true & ~(C["fam_top1"] == HAZ) & sets[:, HAZ], "hidden hazard caught by family set", 2),
        (~single & ~haz_true, "mixed set, human review", 1),
        (U >= tau_u, "high uncertainty, human review", 1),
    ]:
        for i in pick(mask & ~np.isin(np.arange(n), wanted), k):
            wanted.append(int(i)); labels.append(tag)
    wanted, labels = wanted[:8], labels[:8]

    def route(i):
        s = sets[i]
        fams = [FAMILIES[f] for f in np.where(s)[0]]
        if s[HAZ]:
            if size[i] == 1 and U[i] < tau_u:
                return fams, "HAZARDOUS BIN", "#c0392b"
            return fams, "PRIORITY REVIEW (kept out of recycling)", "#e67e22"
        if size[i] == 1 and U[i] < tau_u:
            return fams, f"AUTO: {fams[0]} bin", "#27ae60"
        return fams, "MANUAL REVIEW", "#2980b9"

    rows = []
    cols = 4
    fig, axes = plt.subplots(2, cols, figsize=(15, 8.4))
    for ax, i in zip(axes.ravel(), wanted):
        ax.imshow(Image.open(paths[i]).convert("RGB")); ax.axis("off"); ax.grid(False)
        fams, action, colour = route(i)
        names = C["names"]
        txt = (f"true: {_short(names[C['y'][i]])}\nmodel top-1: {_short(names[top1[i]])}\n"
               f"family set: {{{', '.join(fams)}}}\nuncertainty U = {U[i]:.2f}")
        ax.set_title(txt, fontsize=8.5, loc="left")
        ax.text(0.5, -0.04, action, transform=ax.transAxes, ha="center", va="top", fontsize=9.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.35", fc=colour, ec="none"))
        rows.append({"path": str(paths[i]), "true": names[C["y"][i]], "top1": names[top1[i]], "set": fams,
                     "U": float(U[i]), "action": action})
    for ax in axes.ravel()[len(wanted):]:
        ax.axis("off")
    fig.suptitle("Routing decisions on held-out images (calibration on a separate half; hazard level 0.05, review budget 10%)", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), h_pad=3.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "routing_demo.png", dpi=130); plt.close(fig)

    # worked example for the equations: the first 'hidden hazard caught by family set' item
    hid = [i for i, t in zip(wanted, labels) if t.startswith("hidden")]
    w = {}
    if hid:
        i = hid[0]
        order = np.argsort(-C["probs"][i])[:3]
        w = {
            "path": str(paths[i]), "true": C["names"][C["y"][i]], "top1": C["names"][top1[i]],
            "top3": [(C["names"][j], float(C["probs"][i][j])) for j in order],
            "family_mass": {FAMILIES[f]: float(C["fam_mass"][i][f]) for f in range(len(FAMILIES))},
            "thresholds_q": {FAMILIES[f]: float(q[f]) for f in range(len(FAMILIES))},
            "set": [FAMILIES[f] for f in np.where(sets[i])[0]],
            "U": float(U[i]), "tau_u": tau_u, "attn_convnext_vit": [float(a) for a in raw["attn"][i]],
        }
    w["demo_rows"] = rows
    w["tau_u"] = tau_u
    w["n_calibration"] = int(len(cal)); w["n_demo_pool"] = int(len(ev))
    w["hazard_q"] = float(q[HAZ])
    # quantile threshold example (eq. 19)
    w["eq19"] = {"beta": rs.REVIEW_BUDGET, "tau_u": tau_u}
    # family-accuracy example (eq. 12) on the full test set
    w["family_accuracy_top1"] = float((C["fam_top1"] == C["fam_true"]).mean())
    # attention statistics (eq. 1-2)
    a = raw["attn"].astype(float)
    w["attn_mean"] = [float(a[:, 0].mean()), float(a[:, 1].mean())]
    # sets: typical size (eq. 17)
    w["mean_set_size"] = float(size[ev].mean())
    w["share_hazard_in_set_for_true_hazards"] = float((sets[:, HAZ] & haz_true & is_ev).sum() / (haz_true & is_ev).sum())
    w["n_true_hazards_in_demo_pool"] = int((haz_true & is_ev).sum())
    # eq 3-4 toy using a real item: soda-can item the classifier is unsure about
    names = list(C["names"])
    fit = [names.index(x) for x in ("aluminum_soda_cans", "aluminum_food_cans", "steel_food_cans")]
    cand = np.where((C["y"] == names.index("aluminum_soda_cans")) & (top1 != names.index("aluminum_soda_cans")) & (C["fam_top1"] != HAZ))[0]
    if len(cand):
        i = int(cand[0]); p = C["probs"][i]
        mask = np.zeros(len(p)); mask[fit] = 1
        pm = p * mask / (p * mask).sum()
        c = 0.8
        fused = c * pm + (1 - c) * p
        w["eq34"] = {"true": names[C["y"][i]], "top1_before": names[int(p.argmax())], "p_top1_before": float(p.max()),
                     "p_soda_before": float(p[fit[0]]), "c": c, "p_soda_masked": float(pm[fit[0]]),
                     "p_soda_fused": float(fused[fit[0]]), "top1_after": names[int(fused.argmax())],
                     "p_top1_after": float(fused.max()), "mass_on_fit_before": float(p[fit].sum())}
    (OUT / "worked_example.json").write_text(json.dumps(w, indent=1, default=float))
    return w


def explain_panel() -> None:
    from PIL import Image
    plt = rs._plt()
    d = OUT / "explain_battery"
    g, l = Image.open(d / "gradcam.png").convert("RGB"), Image.open(d / "lime.png").convert("RGB")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    ax[0].imshow(g); ax[0].set_title("Gradient-weighted Class Activation Mapping", fontsize=9)
    ax[1].imshow(l); ax[1].set_title("Local Interpretable Model-agnostic Explanations", fontsize=9)
    for a in ax[:2]:
        a.axis("off"); a.grid(False)
    base, moist, gas = 0.666, -0.693, -0.333
    ax[2].barh(["base value", "moisture sensor (SIMULATED)", "gas sensor (SIMULATED)"][::-1], [base, moist, gas][::-1],
               color=["#e45756" if v < 0 else "#4c78a8" for v in [base, moist, gas][::-1]])
    ax[2].axvline(0, color="k", lw=0.8)
    ax[2].set_title("Shapley Additive Explanations of the\ncontamination score (simulated sensors)", fontsize=9)
    ax[2].set_xlabel("contribution to the score")
    fig.tight_layout(); fig.savefig(OUT / "explain_panel.png", dpi=130); plt.close(fig)


def video_frames() -> None:
    import cv2
    plt = rs._plt()
    cap = cv2.VideoCapture(str(OUT / "video" / "annotated.mp4"))
    wanted = [30, 70, 110, 150, 190, 230]
    fr = {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            fr[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, k in zip(axes.ravel(), wanted):
        ax.imshow(fr[k]); ax.axis("off"); ax.grid(False); ax.set_title(f"frame {k}", fontsize=9)
    fig.suptitle("Video survey: annotated frames (boxes carry the item's track number, object, material and routing)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "video" / "video_frames.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    w = routing_demo()
    explain_panel()
    video_frames()
    print(json.dumps({k: v for k, v in w.items() if k != "demo_rows"}, indent=1, default=float))
    for r in w["demo_rows"]:
        print(r["true"], "|", r["top1"], "|", r["set"], "|", round(r["U"], 2), "|", r["action"])
