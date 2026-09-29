# evaluate_conformal_routing.py - measures hazard leakage of conformal routing against top-1 and confidence-gate baselines
"""Measure the conformal risk-bounded router against top-1 routing on real
held-out data.

The held-out test split is repeatedly halved at random: one half calibrates
the per-family thresholds, the other half is routed and scored. Neither half
was seen in training. Two baselines are scored on the same evaluation halves:

  * argmax      - current behaviour with no gate: route by top-1 class.
  * conf-gate   - top-1 routing, sent to review when max softmax < tau, with
                  tau tuned on the calibration half to give the SAME review
                  workload as the conformal router. This is the fair
                  comparison: equal human effort, who leaks fewer hazards?

Usage:
    python scripts/evaluate_conformal_routing.py --config configs/classifier_convnext_swin.yaml \
        --ckpt runs/stage1_swin/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edgewaste.decision_engine.conformal_routing import (HAZ, ROUTE_AUTO, ROUTE_HAZARD, ConformalRouter,  # noqa: E402
                                 alpha_vector, calibrate, conformal_rank, family_mass, family_matrix,
                                 prediction_sets, route_sets)
from edgewaste.taxonomy import FAMILIES  # noqa: E402


def compute_probs(cfg_path: str, ckpt: str, cache: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["probs"], d["labels"], list(d["class_names"])

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from edgewaste.config import Config
    from edgewaste.data.waste_image_dataset import WasteDataset
    from edgewaste.classification.evaluate_classifier import _load_model
    from edgewaste.common_utils import pick_device

    cfg = Config.load(cfg_path)
    device = pick_device()
    model, class_names = _load_model(Path(ckpt), cfg, device)
    ds = WasteDataset(cfg.data.manifest, "test", cfg.model.image_size, train=False)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers)
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
            labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    np.savez(cache, probs=probs, labels=labels, class_names=np.array(class_names))
    return probs, labels, list(class_names)


def score(kind: np.ndarray, fam_routed: np.ndarray, fam_true: np.ndarray) -> dict:
    """kind: route per item; fam_routed: family gate actually actuated (-1 = none)."""
    auto = (kind == ROUTE_AUTO) | (kind == ROUTE_HAZARD)
    haz = fam_true == HAZ
    wrong_auto = auto & (fam_routed != fam_true)
    return {
        "hazard_leak": float((haz & auto & (fam_routed != HAZ)).sum() / max(haz.sum(), 1)),
        "false_hazard_auto": float((~haz & (kind == ROUTE_HAZARD)).sum() / max((~haz).sum(), 1)),
        "auto_rate": float(auto.mean()),
        "review_rate": float(1 - auto.mean()),
        "misroute_rate": float(wrong_auto.mean()),
        "accuracy_of_auto": float(1 - wrong_auto.sum() / max(auto.sum(), 1)),
    }


def top1_routes(fam_pred: np.ndarray, conf: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    kind = np.full(len(fam_pred), "review", dtype=object)
    ok = conf >= tau
    kind[ok & (fam_pred == HAZ)] = ROUTE_HAZARD
    kind[ok & (fam_pred != HAZ)] = ROUTE_AUTO
    kind[~ok & (fam_pred == HAZ)] = "hazard_review"
    return kind, np.where(ok, fam_pred, -1)


def synthetic_check(n: int, alpha: float, delta: float, rng, trials: int = 20000) -> dict:
    """Validate the guarantee where true coverage is computable exactly:
    scores ~ Uniform(0,1), so the true coverage of threshold q is q itself.
    Real-data halves can't do this — their coverage estimate carries its own
    sampling noise — so this is the check of the math, the real-data run is
    the check of the benefit."""
    k_std = conformal_rank(n, alpha)
    k_pac = conformal_rank(n, alpha, delta)
    s = np.sort(rng.random((trials, n)), axis=1)
    return {"n": n, "alpha": alpha, "delta": delta, "rank_standard": k_std, "rank_pac": k_pac,
            "standard_violation_rate": float((s[:, k_std - 1] < 1 - alpha).mean()),
            "pac_violation_rate": float((s[:, k_pac - 1] < 1 - alpha).mean())}


def tau_for_review_rate(conf: np.ndarray, target: float) -> float:
    return float(np.quantile(conf, target)) if target > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/classifier_convnext_swin.yaml")
    ap.add_argument("--ckpt", default="runs/stage1_swin/best.pt")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = Path(args.ckpt).parent
    probs, labels, class_names = compute_probs(args.config, args.ckpt, out_dir / "probs_test.npz")
    M = family_matrix(class_names)
    fm = family_mass(probs, class_names)
    fam_true = M[labels].argmax(axis=1)
    item_pred = probs.argmax(axis=1)
    fam_pred_top1 = M[item_pred].argmax(axis=1)
    conf = probs.max(axis=1)
    n = len(labels)
    print(f"{n} held-out test items, {int((fam_true == HAZ).sum())} hazardous")

    argmax_all = score(*top1_routes(fam_pred_top1, conf, 0.0), fam_true)

    rng = np.random.default_rng(args.seed)
    n_haz_cal = int((fam_true == HAZ).sum()) // 2
    results = {"n_test": n, "n_hazardous": int((fam_true == HAZ).sum()),
               "trials": args.trials, "argmax_full_test": argmax_all,
               "synthetic_guarantee_check": synthetic_check(n_haz_cal, 0.02, 0.05, rng),
               "configs": []}

    configs = [(0.10, 0.05, None), (0.05, 0.02, None), (0.05, 0.01, None), (0.05, 0.02, 0.05)]
    for alpha, alpha_h, delta in configs:
        alphas = alpha_vector(alpha, alpha_h)
        conf_runs, base_runs, cover = [], [], []
        for _ in range(args.trials):
            perm = rng.permutation(n)
            cal, ev = perm[: n // 2], perm[n // 2:]
            q = calibrate(fm[cal], fam_true[cal], alphas, delta)

            sets_ev = prediction_sets(fm[ev], q)
            kind, fam_r = route_sets(sets_ev)
            conf_runs.append(score(kind, fam_r, fam_true[ev]))
            cover.append([sets_ev[fam_true[ev] == f, f].mean() for f in range(len(FAMILIES))])

            # Baseline at matched workload: tau chosen on the calibration half.
            cal_kind, _ = route_sets(prediction_sets(fm[cal], q))
            target_review = float(np.isin(cal_kind, [ROUTE_AUTO, ROUTE_HAZARD], invert=True).mean())
            tau = tau_for_review_rate(conf[cal], target_review)
            base_runs.append(score(*top1_routes(fam_pred_top1[ev], conf[ev], tau), fam_true[ev]))

        def agg(runs):
            return {k: {"mean": float(np.mean([r[k] for r in runs])),
                        "std": float(np.std([r[k] for r in runs])),
                        "p95": float(np.percentile([r[k] for r in runs], 95))} for k in runs[0]}

        cov = np.array(cover)
        entry = {"alpha": alpha, "alpha_hazard": alpha_h, "delta": delta,
                 "conformal": agg(conf_runs), "conf_gate_matched": agg(base_runs),
                 "coverage_mean_per_family": dict(zip(FAMILIES, cov.mean(0).round(4).tolist())),
                 "hazard_empirical_coverage_below_target_rate": float((cov[:, HAZ] < 1 - alpha_h).mean())}
        results["configs"].append(entry)

    # Deployable thresholds, calibrated on the full test split at the default operating point.
    router = ConformalRouter.fit(probs, labels, class_names, alpha=0.05, alpha_hazard=0.02, delta=0.05)
    router.save(out_dir / "conformal_router.json")

    s = results["synthetic_guarantee_check"]
    print(f"\nsynthetic check (n={n_haz_cal}, alpha=0.02, delta=0.05, exact coverage): "
          f"standard violates {s['standard_violation_rate']:.3f}, PAC violates {s['pac_violation_rate']:.3f}")
    print(f"argmax (no gate): hazard leak {argmax_all['hazard_leak']:.3f}, "
          f"misroute {argmax_all['misroute_rate']:.3f}, auto {argmax_all['auto_rate']:.3f}")
    hdr = (f"{'alpha':>6} {'a_haz':>6} {'delta':>6} | {'method':<10} {'leak mean':>9} {'leak p95':>9} "
           f"{'misroute':>9} {'auto':>7} {'false-haz':>9}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for e in results["configs"]:
        d = "-" if e["delta"] is None else f"{e['delta']:.2f}"
        for name, key in (("conformal", "conformal"), ("conf-gate", "conf_gate_matched")):
            m = e[key]
            print(f"{e['alpha']:>6.2f} {e['alpha_hazard']:>6.2f} {d:>6} | {name:<10} "
                  f"{m['hazard_leak']['mean']:>9.4f} {m['hazard_leak']['p95']:>9.4f} "
                  f"{m['misroute_rate']['mean']:>9.4f} {m['auto_rate']['mean']:>7.3f} "
                  f"{m['false_hazard_auto']['mean']:>9.4f}")

    (out_dir / "conformal_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_dir / 'conformal_results.json'} and {out_dir / 'conformal_router.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
