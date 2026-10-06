# prototype_sorter.py - runs the 2-day hardware prototype: camera + sensors + tilt tray, using the project's own classifier and decision logic
"""One item at a time: put an item on the tray, press the button (or SPACE in the window), and the unit
  1. photographs the item and classifies it (hybrid ConvNeXt + Vision Transformer, 33 classes),
  2. measures uncertainty (Monte Carlo dropout) and builds the calibrated set of plausible families (conformal),
  3. reads the moisture and gas sensors and computes the Organic Contamination Index (if calibrated),
  4. decides with the project's decision engine and moves the tray / lights the leds.

  python scripts/prototype_sorter.py --port COM5 --camera 0                 # real hardware
  python scripts/prototype_sorter.py --mock-node --image some_photo.jpg      # no hardware at all: one decision from an image file
  python scripts/prototype_sorter.py --mock-node --camera 0                  # laptop camera, printed actions
  --camera can also be a phone-camera url such as http://192.168.1.5:8080/video

Keys in the window: SPACE classify now, A toggle auto mode (classifies every 4 s), Q quit.
Thresholds are calibrated at start-up on the validation set of the verified first model (runs/stage1), never on the test set.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgewaste.applications.prototype_node import ACTION_TEXT, MockNode, Node  # noqa: E402
from edgewaste.classification.run_inference import load_for_inference  # noqa: E402
from edgewaste.common_utils import pick_device  # noqa: E402
from edgewaste.config import Config  # noqa: E402
from edgewaste.contamination.oci_model import OCIModel, compute_oci  # noqa: E402
from edgewaste.contamination.sensor_normalization import CalibrationAnchors, normalize_gas, normalize_moisture  # noqa: E402
from edgewaste.decision_engine.conformal_routing import alpha_vector, calibrate, family_matrix, prediction_sets  # noqa: E402
from edgewaste.decision_engine.routing_rules import decide  # noqa: E402
from edgewaste.taxonomy import FAMILIES  # noqa: E402

CACHE_VAL = ROOT / "runs/analysis/convnext_vit/cache_val.npz"
N_MC = 25
ROUTE_TO_CODE = {"hazardous": "H", "manual_review": "V", "contaminated_reject": "C"}      # any family route -> "r"


class Sorter:
    def __init__(self, args):
        self.args = args
        self.dev = pick_device()
        cfg = Config.load(args.config)
        cfg.model.pretrained = False                    # the checkpoint holds every weight; no download needed
        self.model, self.names, self.tfm = load_for_inference(args.ckpt, cfg, self.dev)
        self.head = self.model.head
        self._calibrate_thresholds()
        self.oci = self._load_oci(args.oci)
        self.node = MockNode(wet=args.mock_wet) if args.mock_node else Node(args.port)
        self.log_path = Path(args.log)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        Path(args.snap_dir).mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["time", "item_label", "top1", "confidence", "uncertainty", "family_set", "moisture_raw", "gas_raw", "oci", "route", "command", "reason", "snapshot"])

    # -------------------------------------------------------------- start-up calibration (validation set only)
    @torch.no_grad()
    def _mc(self, emb: torch.Tensor, passes: int = N_MC):
        self.head.train()
        probs = torch.stack([torch.softmax(self.head(emb), 1) for _ in range(passes)]).mean(0)
        self.head.eval()
        ent = -(probs * (probs + 1e-12).log()).sum(1) / np.log(probs.shape[1])
        return probs, ent

    def _calibrate_thresholds(self):
        a = self.args
        d = np.load(CACHE_VAL, allow_pickle=True)
        assert list(d["class_names"]) == list(self.names), "cache and checkpoint disagree on the class order"
        probs = torch.softmax(torch.from_numpy(d["logits"].astype(np.float32)), 1).numpy().astype(np.float64)
        M = family_matrix(list(self.names))
        fam_true = M[d["labels"].astype(int)].argmax(1)
        self.M = M
        self.q = calibrate(probs @ M, fam_true, alpha_vector(a.alpha, a.alpha_h))
        torch.manual_seed(0)
        _, U = self._mc(torch.from_numpy(d["emb"].astype(np.float32)).to(self.dev))
        self.tau_u = float(np.quantile(U.cpu().numpy(), 1 - a.review_budget))
        print(f"calibrated on {len(fam_true)} validation items: alpha={a.alpha}, hazard alpha={a.alpha_h}, review budget={a.review_budget:.0%}, uncertainty threshold={self.tau_u:.3f}")
        print("  family thresholds q_f:", {f: round(float(v), 3) for f, v in zip(FAMILIES, self.q)})

    def _load_oci(self, path):
        p = Path(path)
        if not p.exists():
            print(f"no contamination calibration at {p}: contamination index is OFF (run scripts/calibrate_prototype_oci.py)")
            return None
        j = json.loads(p.read_text())
        print(f"contamination calibration loaded ({j['n_samples']} samples)")
        return {"anchors": CalibrationAnchors(**j["anchors"]), "model": OCIModel(**j["model"]), "thr": {k: v["operating_threshold"] for k, v in j["thresholds"].items()}}

    # -------------------------------------------------------------- one decision
    @torch.no_grad()
    def classify(self, img: Image.Image):
        x = self.tfm(img.convert("RGB")).unsqueeze(0).to(self.dev)
        self.model.eval()
        emb = self.model.feature_vector(x)
        probs_det = torch.softmax(self.head(emb), 1)[0].cpu().numpy().astype(np.float64)
        _, U = self._mc(emb)
        fam_mass = probs_det @ self.M
        sets = prediction_sets(fam_mass[None, :], self.q)[0]
        top = int(probs_det.argmax())
        return self.names[top], float(probs_det[top]), float(U[0]), {FAMILIES[i] for i in np.where(sets)[0]}, probs_det

    def contamination(self):
        """(moisture_raw, gas_raw, oci, threshold) or (..., None, 0.5) when sensors or calibration are missing."""
        r = self.node.sensors()
        if r is None:
            return None, None, None, 0.5
        m_raw, g_raw = r
        if self.oci is None:
            return m_raw, g_raw, None, 0.5
        an = self.oci["anchors"]
        ok_m = 5 < m_raw < 1018
        ok_g = 5 < g_raw < 1018
        f_m = normalize_moisture(m_raw, an) if ok_m else None
        f_g = normalize_gas(1023.0 / g_raw - 1.0, an) if ok_g else None
        if f_m is None and f_g is None:
            return m_raw, g_raw, None, 0.5
        key = "both" if (f_m is not None and f_g is not None) else ("moisture_only" if f_m is not None else "gas_only")
        return m_raw, g_raw, compute_oci(self.oci["model"], f_m, f_g), self.oci["thr"][key]

    def run_once(self, img: Image.Image, label: str = ""):
        name, conf, U, fam_set, probs = self.classify(img)
        m_raw, g_raw, oci, oci_thr = self.contamination()
        dec = decide(name, conf, U, oci, uncertainty_threshold=self.tau_u, oci_threshold=oci_thr, family_set=fam_set)
        code = ROUTE_TO_CODE.get(dec.route, "R")
        snap = Path(self.args.snap_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_{dec.route}.jpg"
        img.save(snap)
        self.node.act(code)
        order = np.argsort(-probs)[:3]
        print("\n  top-3: " + ", ".join(f"{self.names[i]} {probs[i] * 100:.0f}%" for i in order))
        print(f"  uncertainty {U:.2f} (threshold {self.tau_u:.2f})   family set {sorted(fam_set)}")
        print(f"  sensors: moisture {m_raw}  gas {g_raw}  OCI {'-' if oci is None else f'{oci:.2f} (threshold {oci_thr:.2f})'}")
        print(f"  DECISION: {dec.route.upper()}  ->  {ACTION_TEXT[code]}\n  reason: {dec.reason}")
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), label, name, f"{conf:.3f}", f"{U:.3f}", "|".join(sorted(fam_set)), m_raw, g_raw, "" if oci is None else f"{oci:.3f}", dec.route, code, dec.reason, snap.name])
        return dec, name, conf, U, oci

    # -------------------------------------------------------------- camera loop
    def grab_roi(self, frame):
        h, w = frame.shape[:2]
        x, y, rw, rh = self.args.roi
        x0, y0, x1, y1 = int(x * w), int(y * h), int((x + rw) * w), int((y + rh) * h)
        crop = frame[y0:y1, x0:x1]
        s = min(crop.shape[:2])
        cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
        sq = crop[cy - s // 2: cy - s // 2 + s, cx - s // 2: cx - s // 2 + s]
        import cv2
        return Image.fromarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)), (x0, y0, x1, y1)

    def loop(self):
        import cv2
        src = self.args.camera
        cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        if not cap.isOpened():
            raise SystemExit(f"cannot open camera {src}")
        last = ["press SPACE or the button to classify"]
        auto, t_auto = False, 0.0
        print("window open: SPACE = classify, A = auto mode, Q = quit")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera frame missing"); break
            img, box = self.grab_roi(frame)
            show = frame.copy()
            cv2.rectangle(show, box[:2], box[2:], (0, 255, 0), 2)
            for i, t in enumerate(last):
                cv2.putText(show, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(show, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("edge-waste prototype", show)
            k = cv2.waitKey(1) & 0xFF
            trigger = k == ord(" ") or self.node.button() or (auto and time.time() - t_auto > 4)
            if k in (ord("q"), 27):
                break
            if k == ord("a"):
                auto = not auto; t_auto = time.time(); last = [f"auto mode {'ON' if auto else 'OFF'}"]
            if trigger:
                t_auto = time.time()
                dec, name, conf, U, oci = self.run_once(img)
                last = [f"{name} {conf * 100:.0f}%   U={U:.2f}", f"-> {dec.route}" + ("" if oci is None else f"   OCI={oci:.2f}")]
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM5", help="serial port of the Arduino (Device Manager shows it)")
    ap.add_argument("--mock-node", action="store_true", help="no hardware: print actions, invent sensor values")
    ap.add_argument("--mock-wet", action="store_true", help="with --mock-node: the invented sensors look contaminated")
    ap.add_argument("--camera", default="0", help="camera index or a stream url")
    ap.add_argument("--image", help="classify this image once instead of opening the camera")
    ap.add_argument("--label", default="", help="with --image: the true item name, written to the log")
    ap.add_argument("--roi", type=lambda s: [float(v) for v in s.split(",")], default=[0.15, 0.1, 0.7, 0.8], help="x,y,w,h of the tray area as fractions of the frame")
    ap.add_argument("--ckpt", default=str(ROOT / "runs/stage1/best.pt"))
    ap.add_argument("--config", default=str(ROOT / "configs/classifier_convnext_vit.yaml"))
    ap.add_argument("--oci", default=str(ROOT / "runs/prototype/oci_calibration.json"))
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--alpha-h", type=float, default=0.05, dest="alpha_h")
    ap.add_argument("--review-budget", type=float, default=0.10, dest="review_budget")
    ap.add_argument("--log", default=str(ROOT / "runs/prototype/log.csv"))
    ap.add_argument("--snap-dir", default=str(ROOT / "runs/prototype/snaps"), dest="snap_dir")
    a = ap.parse_args()
    s = Sorter(a)
    if a.image:
        s.run_once(Image.open(a.image), a.label)
    else:
        s.loop()


if __name__ == "__main__":
    main()
