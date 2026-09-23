"""Combined explainability demo: Grad-CAM + LIME for the classifier, SHAP
for OCI, on one real image.

Grad-CAM and LIME answer different questions (see explain_lime.py's module
docstring) and are shown side by side deliberately, not merged into one
number — agreement between them is reassuring, disagreement is itself a
finding worth reporting, and averaging them away would hide that.

Usage:
    python scripts/explain_demo.py path/to/image.jpg
    python scripts/explain_demo.py path/to/image.jpg --out-dir runs/explain_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from edgewaste.config import Config
from edgewaste.explain import build_gradcam, gradcam_overlay
from edgewaste.explain_lime import lime_explain_crop, lime_feature_weights
from edgewaste.infer import load_for_inference
from edgewaste.oci.explain_shap import explain_one, shap_explain_oci
from edgewaste.sensors import fit_demo_oci_model, reading_to_features, simulate_sensors
from edgewaste.utils import pick_device


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grad-CAM + LIME + SHAP explanation for one image.")
    ap.add_argument("image", help="Path to the item image/crop.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--ckpt", default="runs/stage1/best.pt")
    ap.add_argument("--out-dir", default="runs/explain_demo")
    ap.add_argument("--lime-samples", type=int, default=500)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config.load(args.config)
    device = pick_device()
    classifier, class_names, tfm = load_for_inference(args.ckpt, cfg, device)
    class_names = list(class_names)

    crop = Image.open(args.image).convert("RGB")

    print(f"=== Classifying {args.image} ===")
    x = tfm(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(classifier(x), dim=1)[0]
    conf, idx = probs.max(0)
    pred_name = class_names[int(idx)]
    print(f"Prediction: {pred_name}  ({float(conf)*100:.1f}%)\n")

    print("=== Grad-CAM (ConvNeXt branch) ===")
    cam = build_gradcam(classifier)
    _, gradcam_rgb = gradcam_overlay(cam, classifier, tfm, crop, device, class_idx=int(idx))
    gradcam_path = out / "gradcam.png"
    Image.fromarray(gradcam_rgb).save(gradcam_path)
    print(f"Saved {gradcam_path}\n")

    print(f"=== LIME ({args.lime_samples} perturbation samples) ===")
    lime_idx, lime_name, explanation, lime_overlay = lime_explain_crop(
        classifier, tfm, crop, device, class_names, num_samples=args.lime_samples)
    lime_path = out / "lime.png"
    Image.fromarray(lime_overlay).save(lime_path)
    weights = sorted(lime_feature_weights(explanation, lime_idx), key=lambda w: -abs(w[1]))
    print(f"LIME's own top prediction: {lime_name}")
    print(f"Top contributing superpixels (id, weight): {weights[:5]}")
    print(f"Saved {lime_path}")
    if lime_name != pred_name:
        print(f"NOTE: LIME's top label ({lime_name}) differs from the "
              f"classifier's reported argmax ({pred_name}) — a genuine "
              f"disagreement between the two explanation methods, not a bug.")
    print()

    print("=== OCI + SHAP (simulated sensors, since no physical hardware exists) ===")
    oci_model = fit_demo_oci_model()
    reading = simulate_sensors(pred_name, rng=np.random.default_rng(0))
    f_m, f_g = reading_to_features(reading)
    rng = np.random.default_rng(1)
    f_m_bg = rng.uniform(0, 1, 200)
    f_g_bg = rng.uniform(0, 1, 200)
    print(explain_one(oci_model, f_m, f_g, f_m_bg, f_g_bg))

    print(f"\nAll artefacts written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
