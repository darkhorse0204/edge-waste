# make_demo_gallery.py - builds the wider set of demonstration figures, animations and short videos for the invention disclosure draft
"""Demonstration gallery: every class, every stage.

Run one part at a time (keeps memory low on this machine):
  python scripts/make_demo_gallery.py classes     # all 33 classes with routing (3 figures) + per-class F1 chart
  python scripts/make_demo_gallery.py hazards     # hazardous-family gallery with family sets
  python scripts/make_demo_gallery.py gradcam     # Grad-CAM heat maps for 16 classes
  python scripts/make_demo_gallery.py detector    # 18-class object detector on benchmark images + mask cut-outs
  python scripts/make_demo_gallery.py videos      # class-tour video/animation and video-survey filmstrip, inventory charts
  python scripts/make_demo_gallery.py oci         # simulated conveyor with a sensor failure
  python scripts/make_demo_gallery.py federated   # federated averaging simulation on stored embeddings
  python scripts/make_demo_gallery.py gan         # real versus generated images

Outputs go to reports/demo/gallery/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("reports/demo/gallery")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 7
NL = chr(10)


def _short(s: str) -> str:
    return s.replace("_", " ")


# ---------------------------------------------------------------------------
# shared routing set-up (same calibration half and thresholds as make_demo_snapshots.py)
# ---------------------------------------------------------------------------
def setup_routing():
    import run_simulations as rs
    from edgewaste.decision_engine.conformal_routing import HAZ, alpha_vector, calibrate, prediction_sets
    from edgewaste.taxonomy import FAMILIES
    C = rs.load_clean()
    raw = np.load(rs.CACHE, allow_pickle=True)
    head, _ = rs.load_head()
    _, U = rs.mc_uncertainty(head, C["emb"])
    U = U.numpy()
    n = len(C["y"])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    cal, ev = perm[: n // 2], perm[n // 2:]
    q = calibrate(C["fam_mass"][cal], C["fam_true"][cal], alpha_vector(0.10, 0.05))
    tau_u = float(np.quantile(U[cal], 1 - rs.REVIEW_BUDGET))
    sets = prediction_sets(C["fam_mass"], q)
    return dict(C=C, paths=raw["paths"], U=U, ev=ev, cal=cal, q=q, tau_u=tau_u, sets=sets, HAZ=HAZ,
                FAMILIES=FAMILIES, top1=C["probs"].argmax(1), conf=C["probs"].max(1), rs=rs)


def decide(R, i):
    s = R["sets"][i]; HAZ = R["HAZ"]
    fams = [R["FAMILIES"][f] for f in np.where(s)[0]]
    ok = s.sum() == 1 and R["U"][i] < R["tau_u"]
    if s[HAZ]:
        return fams, ("HAZARDOUS BIN", "#c0392b") if ok else ("PRIORITY REVIEW", "#e67e22")
    if ok:
        return fams, (f"AUTO: {fams[0]} bin", "#27ae60")
    return fams, ("MANUAL REVIEW", "#2980b9")


def draw_panels(R, idx, ncols, title, out, figw=15.0, rowh=3.9, fs=8.5):
    from PIL import Image
    plt = R["rs"]._plt()
    C = R["C"]; names = C["names"]
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figw, rowh * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    rows = []
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(Image.open(R["paths"][i]).convert("RGB")); ax.grid(False)
        fams, (act, col) = decide(R, i)
        right = C["y"][i] == R["top1"][i]
        ax.set_title(f"true: {_short(names[C['y'][i]])}{NL}model: {_short(names[R['top1'][i]])} ({R['conf'][i] * 100:.0f}%){NL}"
                     f"family set: {{{', '.join(fams)}}}   U = {R['U'][i]:.2f}", fontsize=fs, loc="left",
                     color="black" if right else "#a04000")
        ax.text(0.5, -0.03, act, transform=ax.transAxes, ha="center", va="top", fontsize=fs + 0.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="none"))
        rows.append(dict(true=names[C["y"][i]], model=names[R["top1"][i]], set=fams, U=float(R["U"][i]), action=act))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97), h_pad=2.6)
    fig.savefig(out, dpi=115); plt.close(fig)
    return rows


def part_classes():
    R = setup_routing(); C = R["C"]
    rng = np.random.default_rng(SEED + 1)
    pick = []
    for c in range(len(C["names"])):
        pool = [i for i in R["ev"] if C["y"][i] == c]
        pick.append(int(rng.choice(pool)))
    meta = {}
    for k, (a, b) in enumerate([(0, 11), (11, 22), (22, 33)]):
        meta[k] = draw_panels(R, pick[a:b], 3,
                              f"All 33 classes, one random held-out image each ({a + 1} to {b}); orange titles mark a wrong top-1 class",
                              OUT / f"classes_{k + 1}.png", figw=11.0, rowh=3.5, fs=9.5)
    wrong = sum(C["y"][i] != R["top1"][i] for i in pick)
    # per-class f1 bar chart
    import pandas as pd
    plt = R["rs"]._plt()
    df = pd.read_csv("reports/ml_analysis/per_class_metrics_test.csv")
    fam_col = {f: c for f, c in zip(R["FAMILIES"], plt.cm.tab10(np.arange(10)))}
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(range(len(df)), df.f1 * 100, color=[fam_col[f] for f in df.family])
    ax.set_xticks(range(len(df)), [_short(c) for c in df["class"]], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("F1 score (%)"); ax.set_ylim(0, 105)
    for i, v in enumerate(df.f1 * 100):
        ax.text(i, v + 0.8, f"{v:.0f}", ha="center", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=fam_col[f]) for f in R["FAMILIES"]]
    ax.legend(handles, R["FAMILIES"], ncol=9, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(); fig.savefig(OUT / "per_class_f1.png", dpi=125); plt.close(fig)
    wrong_info = []
    for i in pick:
        if C["y"][i] != R["top1"][i]:
            wrong_info.append({"true": C["names"][C["y"][i]], "model": C["names"][R["top1"][i]], "same_family": bool(C["fam_true"][i] == C["fam_top1"][i]),
                               "action": decide(R, i)[1][0]})
    (OUT / "classes_meta.json").write_text(json.dumps({"picked_wrong_top1": int(wrong), "n": len(pick), "wrong": wrong_info,
                                                        "f1_min": float(df.f1.min()), "f1_max": float(df.f1.max()),
                                                        "f1_min_class": df["class"][df.f1.idxmin()], "f1_max_class": df["class"][df.f1.idxmax()]}, indent=1))
    print("classes done; wrong top-1 among the 33 picks:", wrong)


def part_hazards():
    R = setup_routing(); C = R["C"]; HAZ = R["HAZ"]
    rng = np.random.default_rng(SEED + 2)
    names = list(C["names"])
    pick = []
    for cls in ("battery", "e_waste", "medical"):
        c = names.index(cls)
        ok = [i for i in R["ev"] if C["y"][i] == c and R["top1"][i] == c]
        hid = [i for i in R["ev"] if C["y"][i] == c and R["top1"][i] != c and R["sets"][i, HAZ]]
        pick += [int(i) for i in rng.choice(ok, 2, replace=False)]
        pick += [int(i) for i in rng.choice(hid, min(2, len(hid)), replace=False)]
    rows = draw_panels(R, pick, 4, "Hazardous family: confident hazards (first two of each class) and hazards whose top-1 class differed from the true class (the family set still kept them out of recycling)",
                       OUT / "hazards.png")
    # also aerosol cans (a hazardous-leaning class sorted with metal)
    print("hazards done", len(pick))


# ---------------------------------------------------------------------------
def part_gradcam():
    import torch
    from PIL import Image
    import run_simulations as rs
    from edgewaste.classification.run_inference import load_for_inference
    from edgewaste.config import Config
    from edgewaste.explainability.gradcam_explanation import build_gradcam, gradcam_overlay
    from edgewaste.common_utils import pick_device
    cfg = Config.load("configs/classifier_convnext_vit.yaml")
    dev = pick_device()
    model, names, tfm = load_for_inference("runs/stage1/best.pt", cfg, dev)
    cam = build_gradcam(model)
    raw = np.load(rs.CACHE, allow_pickle=True)
    paths, labels = raw["paths"], raw["labels"].astype(int)
    probs = torch.softmax(torch.from_numpy(raw["logits"]), 1).numpy()
    rng = np.random.default_rng(SEED + 3)
    wanted = ["plastic_water_bottles", "plastic_shopping_bags", "newspaper", "cardboard_boxes", "glass_beverage_bottles", "aluminum_soda_cans",
              "steel_food_cans", "food_waste", "coffee_grounds", "styrofoam_cups", "clothing", "shoes", "battery", "e_waste", "medical", "aerosol_cans"]
    plt = rs._plt()
    fig, axes = plt.subplots(4, 4, figsize=(15, 9.2))
    for ax, cls in zip(axes.ravel(), wanted):
        c = names.index(cls)
        pool = [i for i in range(len(labels)) if labels[i] == c and probs[i].argmax() == c]
        i = int(rng.choice(pool))
        img = Image.open(paths[i]).convert("RGB")
        rgb, over = gradcam_overlay(cam, model, tfm, img, dev, class_idx=c)
        both = np.concatenate([rgb, over], axis=1)
        ax.imshow(both); ax.axis("off"); ax.grid(False)
        ax.set_title(f"{_short(cls)}   (confidence {probs[i][c] * 100:.0f}%)", fontsize=9)
    fig.suptitle("Gradient-weighted Class Activation Mapping on 16 classes: each pair shows the model input and the heat map (red = strongest influence)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(OUT / "gradcam_16.png", dpi=105); plt.close(fig)
    print("gradcam done")


# ---------------------------------------------------------------------------
def part_detector():
    from PIL import Image
    import run_simulations as rs
    from ultralytics import YOLO
    plt = rs._plt()
    det = YOLO("runs/detect/runs/detect/taco_multiclass/weights/best.pt")
    imgs = sorted(Path("data/detect/taco/images/val").glob("*.jpg"))
    rng = np.random.default_rng(SEED)
    pick = [imgs[i] for i in rng.choice(len(imgs), 40, replace=False)]
    from edgewaste.detection.sam_segmentation import load_sam, segment_box
    sam = load_sam()
    shown = []
    for p in pick:
        r = det.predict(str(p), conf=0.35, verbose=False, device=0)[0]
        if len(r.boxes) >= 1:
            shown.append((p, r))
        if len(shown) == 12:
            break
    fig, axes = plt.subplots(3, 4, figsize=(15, 11.6))
    for ax, (p, r) in zip(axes.ravel(), shown):
        ax.imshow(r.plot()[:, :, ::-1]); ax.axis("off"); ax.grid(False)
        ax.set_title(f"{len(r.boxes)} item(s) found", fontsize=9)
    fig.suptitle("Eighteen-class object detector on held-out benchmark photographs (box, object name, confidence)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "detector_12.png", dpi=105); plt.close(fig)
    # sam cut-outs of the largest box in 6 of them
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6))
    fills = []
    for k, (p, r) in enumerate(shown[:6]):
        im = Image.open(p).convert("RGB")
        b = r.boxes.xyxy.cpu().numpy()
        a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        box = tuple(int(v) for v in b[a.argmax()])
        seg = segment_box(sam, im, box)
        arr = np.array(im).copy()
        over = arr.copy(); over[seg.mask] = (0.45 * over[seg.mask] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
        cut = arr.copy(); cut[~seg.mask] = 0
        x1, y1, x2, y2 = box
        pad = 10
        sl = (slice(max(y1 - pad, 0), y2 + pad), slice(max(x1 - pad, 0), x2 + pad))
        axes[0, k].imshow(over[sl]); axes[1, k].imshow(cut[sl])
        axes[0, k].set_title(f"{r.names[int(r.boxes.cls[a.argmax()])]}{NL}mask fills {seg.fill_ratio * 100:.0f}% of the box", fontsize=8)
        fills.append(float(seg.fill_ratio))
        for ax in axes[:, k]:
            ax.axis("off"); ax.grid(False)
    fig.suptitle("Segment Anything Model: pixel-accurate item outline inside each detector box (top) and the background-removed cut-out passed on (bottom)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "sam_6.png", dpi=110); plt.close(fig)
    print("detector + sam done", fills)


# ---------------------------------------------------------------------------
def part_videos():
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    R = setup_routing(); C = R["C"]
    rs = R["rs"]; plt = rs._plt()
    # 1. class-tour video: one image per class, prediction and routing overlay
    rng = np.random.default_rng(SEED + 1)
    pick = []
    for c in range(len(C["names"])):
        pool = [i for i in R["ev"] if C["y"][i] == c]
        pick.append(int(rng.choice(pool)))
    W, H = 640, 480
    vw = cv2.VideoWriter(str(OUT / "class_tour.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    frames_gif = []
    strip = []
    try:
        font = ImageFont.truetype("arial.ttf", 20); font_b = ImageFont.truetype("arialbd.ttf", 24)
    except OSError:
        font = font_b = ImageFont.load_default()
    for n, i in enumerate(pick):
        im = Image.open(R["paths"][i]).convert("RGB")
        im.thumbnail((W, H - 110))
        canvas = Image.new("RGB", (W, H), (20, 20, 20))
        canvas.paste(im, ((W - im.width) // 2, 0))
        fams, (act, col) = decide(R, i)
        d = ImageDraw.Draw(canvas)
        d.text((10, H - 106), f"true: {_short(C['names'][C['y'][i]])}   |   model: {_short(C['names'][R['top1'][i]])} ({R['conf'][i] * 100:.0f}%)", fill=(240, 240, 240), font=font)
        d.text((10, H - 78), f"family set: {{{', '.join(fams)}}}   uncertainty U = {R['U'][i]:.2f}", fill=(200, 200, 200), font=font)
        rgb = tuple(int(col[k:k + 2], 16) for k in (1, 3, 5))
        d.rectangle([0, H - 44, W, H], fill=rgb)
        d.text((10, H - 40), act, fill=(255, 255, 255), font=font_b)
        d.text((W - 110, 6), f"class {n + 1}/33", fill=(255, 255, 0), font=font)
        arr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        for _ in range(20):
            vw.write(arr)
        frames_gif.append(canvas.resize((400, 300)))
        if n % 3 == 0:
            strip.append(canvas)
    vw.release()
    frames_gif[0].save(OUT / "class_tour.gif", save_all=True, append_images=frames_gif[1:], duration=1000, loop=0, optimize=True)
    # filmstrip of 12 frames
    fig, axes = plt.subplots(3, 4, figsize=(14, 8.4))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, f in zip(axes.ravel(), strip):
        ax.imshow(f); ax.grid(False)
    fig.suptitle("Class-tour video: every third class of the 33-class tour (the full video is class_tour.mp4)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "class_tour_filmstrip.png", dpi=110); plt.close(fig)

    # 2. survey video: one frame per scene (12 scenes), gif, inventory charts
    cap = cv2.VideoCapture("reports/demo/video/annotated.mp4")
    fr = []; i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); i += 1
    sel = [10 + 20 * k for k in range(12)]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.8))
    for ax, k in zip(axes.ravel(), sel):
        ax.imshow(fr[k]); ax.axis("off"); ax.grid(False); ax.set_title(f"scene {sel.index(k) + 1}, frame {k}", fontsize=9)
    fig.suptitle("Video survey: one annotated frame from each of the 12 scenes of the 240-frame test video", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "survey_12_scenes.png", dpi=105); plt.close(fig)
    gif = [Image.fromarray(fr[k]).resize((320, 240)) for k in range(0, len(fr), 3)]
    gif[0].save(OUT / "survey.gif", save_all=True, append_images=gif[1:], duration=200, loop=0, optimize=True)
    import pandas as pd
    t = pd.read_csv("reports/demo/video/tracks.csv")
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    t.object_class.value_counts().plot.barh(ax=ax[0], color="#4c78a8"); ax[0].set_title("entries by object type", fontsize=9)
    t.material.value_counts().plot.barh(ax=ax[1], color="#54a24b"); ax[1].set_title("entries by material", fontsize=9)
    t.route.value_counts().plot.barh(ax=ax[2], color="#f58518"); ax[2].set_title("entries by routing decision", fontsize=9)
    ax[3].bar(range(len(t)), t.frames_visible, color="#b279a2"); ax[3].set_title("frames in which each entry was followed", fontsize=9)
    ax[3].set_xlabel("inventory entry")
    fig.tight_layout(); fig.savefig(OUT / "survey_inventory.png", dpi=120); plt.close(fig)
    print("videos done")


# ---------------------------------------------------------------------------
def part_oci():
    import run_simulations as rs
    from edgewaste.contamination.synthetic_calibration_data import generate_synthetic_calibration_data
    from edgewaste.contamination.sensor_normalization import normalize_gas, normalize_moisture
    from edgewaste.contamination.oci_model import fit_oci_weights, compute_oci, select_threshold
    plt = rs._plt()
    df, anchors = generate_synthetic_calibration_data(n_replicates=300, seed=0)
    print(df.columns.tolist())
    df["fm"] = [normalize_moisture(m, anchors) for m in df.moisture_raw]
    df["fg"] = [normalize_gas(r * 1.0, anchors) if False else float(np.clip((-np.log(r) - anchors.l_min) / (anchors.l_max - anchors.l_min), 0, 1)) for r in df.rs_over_r0]
    ycol = [c for c in df.columns if "contam" in c.lower() or c in ("label", "y")][0]
    y = df[ycol].values.astype(int)
    from sklearn.linear_model import LogisticRegression
    X = np.c_[df.fm, df.fg, df.fm * df.fg]
    both = LogisticRegression(C=10).fit(X, y)
    m_only = LogisticRegression(C=10).fit(df[["fm"]], y)
    g_only = LogisticRegression(C=10).fit(df[["fg"]], y)
    from sklearn.metrics import roc_curve
    def thr(score):
        fpr, tpr, th = roc_curve(y, score); ok = tpr >= 0.95
        return float(th[ok][np.argmin(fpr[ok])])
    t_both = thr(both.predict_proba(X)[:, 1]); t_m = thr(m_only.predict_proba(df[["fm"]])[:, 1]); t_g = thr(g_only.predict_proba(df[["fg"]])[:, 1])
    # a 60-item conveyor stream drawn from a held-out draw
    df2, _ = generate_synthetic_calibration_data(n_replicates=20, seed=11, anchors=anchors)
    df2["fm"] = [normalize_moisture(m, anchors) for m in df2.moisture_raw]
    df2["fg"] = np.clip((-np.log(df2.rs_over_r0) - anchors.l_min) / (anchors.l_max - anchors.l_min), 0, 1)
    y2 = df2[ycol].values.astype(int)
    r3 = np.random.default_rng(3)
    idx = r3.permutation(np.r_[r3.choice(np.where(y2 == 1)[0], 30, replace=False), r3.choice(np.where(y2 == 0)[0], 30, replace=False)])
    s = df2.iloc[idx].reset_index(drop=True); ys = y2[idx]
    score = np.zeros(60); mode = []
    for k in range(60):
        gas_lost = 25 <= k < 40
        if gas_lost:
            score[k] = m_only.predict_proba(s[["fm"]].iloc[[k]])[0, 1]; mode.append("moisture only")
        else:
            score[k] = both.predict_proba(np.c_[s.fm[k], s.fg[k], s.fm[k] * s.fg[k]])[0, 1]; mode.append("both sensors")
    thr_used = np.array([t_m if m == "moisture only" else t_both for m in mode])
    flag = score >= thr_used
    fig, ax = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.2, 0.7]})
    ax[0].plot(s.fm, "-o", ms=3, color="#4c78a8", label="moisture score (0 to 1)")
    gas = s.fg.copy().astype(float); gas[25:40] = np.nan
    ax[0].plot(gas, "-o", ms=3, color="#e45756", label="gas score (0 to 1)")
    ax[0].axvspan(24.5, 39.5, color="#cccccc", alpha=0.5); ax[0].text(32, 1.02, "gas sensor lost", ha="center", fontsize=9)
    ax[0].set_ylim(0, 1.15); ax[0].legend(loc="lower left", fontsize=8, ncol=2); ax[0].set_ylabel("sensor scores")
    ax[0].set_title("SIMULATED conveyor of 60 items: sensor scores, Organic Contamination Index and decisions (the model switches itself when the gas sensor fails)", fontsize=9)
    ax[1].bar(range(60), score, color=np.where(ys == 1, "#e45756", "#54a24b"), alpha=0.85)
    ax[1].step(range(60), thr_used, where="mid", color="k", lw=1.2, label="threshold of the model in use (95% sensitivity)")
    ax[1].set_ylabel("Organic Contamination Index"); ax[1].legend(fontsize=8, loc="upper left")
    ax[1].text(59.5, 1.0, "bar colour: red = truly contaminated, green = clean", ha="right", fontsize=8)
    for k in range(60):
        ax[2].add_patch(plt.Rectangle((k - 0.45, 0.1), 0.9, 0.8, color=("#c0392b" if flag[k] else "#27ae60")))
    ax[2].set_ylim(0, 1); ax[2].set_yticks([]); ax[2].set_xlabel("item number on the conveyor")
    ax[2].set_title("decision: red = flagged contaminated (diverted), green = passed", fontsize=8.5)
    ax[2].grid(False)
    fig.tight_layout(); fig.savefig(OUT / "oci_conveyor.png", dpi=125); plt.close(fig)
    # graded response: organic contamination index against the residue mass left on the item, for the three models (held-out simulated draw)
    levels = sorted(df2.residue_mass_g.unique())
    scores = {"both sensors": both.predict_proba(np.c_[df2.fm, df2.fg, df2.fm * df2.fg])[:, 1],
              "moisture sensor only": m_only.predict_proba(df2[["fm"]])[:, 1], "gas sensor only": g_only.predict_proba(df2[["fg"]])[:, 1]}
    ths = {"both sensors": t_both, "moisture sensor only": t_m, "gas sensor only": t_g}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    resp = {}
    for ax_, (name, sc) in zip(axs, scores.items()):
        data = [sc[df2.residue_mass_g.values == lv] for lv in levels]
        bp = ax_.boxplot(data, positions=range(len(levels)), widths=0.6, patch_artist=True, showfliers=False)
        for k, box in enumerate(bp["boxes"]):
            box.set_facecolor("#e45756" if levels[k] >= 5 else "#54a24b"); box.set_alpha(0.75)
        ax_.axhline(ths[name], color="k", ls="--", lw=1.2); ax_.text(len(levels) - 0.5, ths[name] + 0.02, "threshold (95% sensitivity)", ha="right", fontsize=8)
        ax_.set_xticks(range(len(levels)), [f"{lv:g} g" for lv in levels]); ax_.set_xlabel("residue left on the item")
        ax_.set_title(name, fontsize=9.5)
        resp[name] = [float(np.median(d)) for d in data]
    axs[0].set_ylabel("Organic Contamination Index")
    fig.suptitle("SIMULATED sensors: the index rises with the amount of residue for all three models (green = counted clean, red = counted contaminated)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "oci_levels.png", dpi=125); plt.close(fig)
    (OUT / "oci_levels.json").write_text(json.dumps({"levels_g": [float(l) for l in levels], "median_oci": resp}, indent=1))
    sens = float(flag[ys == 1].mean()); fpr = float(flag[ys == 0].mean())
    (OUT / "oci_meta.json").write_text(json.dumps({"sensitivity": sens, "false_positive_rate": fpr, "n_contaminated": int((ys == 1).sum()),
                                                    "thresholds": {"both": t_both, "moisture_only": t_m, "gas_only": t_g}}, indent=1))
    print("oci done", sens, fpr, t_both, t_m, t_g)


# ---------------------------------------------------------------------------
def part_federated():
    import torch
    import torch.nn as nn
    import run_simulations as rs
    plt = rs._plt()
    tr = np.load("runs/analysis/convnext_vit/cache_train.npz", allow_pickle=True)
    te = np.load("runs/analysis/convnext_vit/cache_test.npz", allow_pickle=True)
    Xtr = torch.from_numpy(tr["emb"].astype(np.float32)); ytr = torch.from_numpy(tr["labels"].astype(np.int64))
    Xte = torch.from_numpy(te["emb"].astype(np.float32)); yte = torch.from_numpy(te["labels"].astype(np.int64))
    K, ROUNDS, LOCAL = 5, 15, 2
    torch.manual_seed(0)
    # non-identical clients: each unit mostly sees a different subset of classes (dirichlet 0.3)
    rng = np.random.default_rng(0)
    idx_by_class = [np.where(ytr.numpy() == c)[0] for c in range(33)]
    parts = [[] for _ in range(K)]
    for c, ids in enumerate(idx_by_class):
        p = rng.dirichlet([0.3] * K); cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(rng.permutation(ids), cuts)):
            parts[k] += chunk.tolist()
    sizes = [len(p) for p in parts]
    mk = lambda: nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 33))

    def train(model, ids, epochs):
        opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-2)
        ids = torch.tensor(ids)
        for _ in range(epochs):
            perm = ids[torch.randperm(len(ids))]
            for b in range(0, len(perm), 128):
                bi = perm[b:b + 128]
                opt.zero_grad(); nn.functional.cross_entropy(model(Xtr[bi]), ytr[bi]).backward(); opt.step()

    acc = lambda m: float((m(Xte).argmax(1) == yte).float().mean())
    g = mk(); curve = []
    for r in range(ROUNDS):
        states = []
        for k in range(K):
            m = mk(); m.load_state_dict(g.state_dict()); train(m, parts[k], LOCAL); states.append(m.state_dict())
        new = {key: sum(states[k][key] * (sizes[k] / sum(sizes)) for k in range(K)) for key in states[0]}   # equation 13
        g.load_state_dict(new); curve.append(acc(g))
    local = []
    for k in range(K):
        m = mk(); train(m, parts[k], LOCAL * ROUNDS); local.append(acc(m))
    cen = mk(); train(cen, list(range(len(ytr))), 12); cen_acc = acc(cen)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(range(1, ROUNDS + 1), np.array(curve) * 100, "-o", color="#4c78a8", label="federated averaging (Equation 13)")
    ax.axhline(cen_acc * 100, color="#54a24b", ls="--", label=f"one central model on all images ({cen_acc * 100:.1f}%)")
    ax.axhline(np.mean(local) * 100, color="#e45756", ls=":", label=f"each unit alone, average ({np.mean(local) * 100:.1f}%)")
    ax.set_xlabel("communication round"); ax.set_ylabel("test accuracy (%) on 4,232 held-out images"); ax.legend(fontsize=8)
    ax.set_title(f"Federated learning simulation: {K} sorting units with different class mixes; only parameters are shared", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "federated.png", dpi=125); plt.close(fig)
    (OUT / "federated_meta.json").write_text(json.dumps({"clients": K, "rounds": ROUNDS, "sizes": sizes, "final": curve[-1], "curve": curve,
                                                          "local_mean": float(np.mean(local)), "local_min": float(np.min(local)), "local_max": float(np.max(local)),
                                                          "central": cen_acc}, indent=1))
    print("federated done", curve[-1], np.mean(local), cen_acc, sizes)


# ---------------------------------------------------------------------------
def part_gan():
    import shutil
    shutil.copy("runs/gan/textile/sample_grid.png", OUT / "gan.png")
    print("gan done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["classes", "hazards", "gradcam", "detector", "videos", "oci", "federated", "gan"])
    a = ap.parse_args()
    globals()[f"part_{a.part}"]()
