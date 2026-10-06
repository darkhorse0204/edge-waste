# calibrate_prototype_oci.py - collects real sensor readings for the prototype and fits the three contamination models (the mini calibration protocol)
"""Mini version of the project's calibration protocol, sized for a 2-day prototype.

  python scripts/calibrate_prototype_oci.py collect --port COM5        # interactive; add --mock to test without hardware
  python scripts/calibrate_prototype_oci.py fit                        # writes runs/prototype/oci_calibration.json

Inside `collect` type one command per line:
  air                  gas baseline in clean air (do this first, after the sensor has warmed up for 5+ minutes)
  smell                gas reference near a strong smell (rotten food, vinegar or a spirit swab, a few cm from the sensor)
  dry                  moisture reference: the blade on a dry clean item
  wet                  moisture reference: the blade on a soaked item (or the sensor tip in a glass of water, up to the line)
  sample <cat> <tsp>   one labelled sample, e.g.  sample porous 2   or  sample rigid 0
                       cat = porous (cardboard, paper) or rigid (plastic, metal); tsp = teaspoons of wet food residue on the item
  show                 list what has been collected
  quit

An item is labelled contaminated when it carries at least --contaminated-from teaspoons (default 1, about 5 g).
Collect at least 8 samples in each class (clean and contaminated), ideally 20 or more: 2 categories x levels 0, 0.5, 1, 2, 4 tsp x 4 repeats.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "prototype"
SAMPLES = OUT / "oci_samples.csv"
REFS = OUT / "oci_references.json"
CALIB = OUT / "oci_calibration.json"
ADC_MAX = 1023


def rs_of(adc: float) -> float:
    """Sensor resistance in units of the load resistor: Rs/RL = ADC_MAX/adc - 1. The load resistor cancels in Rs/R0."""
    return ADC_MAX / adc - 1.0


def read(node, n: int = 5) -> tuple[float, float]:
    ms, gs = [], []
    for _ in range(n):
        r = node.sensors()
        if r is None:
            raise SystemExit("the node stopped answering")
        ms.append(r[0]); gs.append(r[1])
        time.sleep(0.4)
    return float(np.median(ms)), float(np.median(gs))


def collect(args):
    from edgewaste.applications.prototype_node import MockNode, Node
    node = MockNode() if args.mock else Node(args.port)
    OUT.mkdir(parents=True, exist_ok=True)
    refs = json.loads(REFS.read_text()) if REFS.exists() else {}
    new = not SAMPLES.exists()
    f = open(SAMPLES, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["category", "teaspoons", "contaminated", "moisture_raw", "gas_raw", "time"]); f.flush()
    print(__doc__)
    while True:
        try:
            line = input("> ").strip().split()
        except EOFError:
            break
        if not line:
            continue
        c = line[0].lower()
        if c in ("quit", "q", "exit"):
            break
        if c == "show":
            print("references:", refs)
            print(f"samples file: {SAMPLES}  ({sum(1 for _ in open(SAMPLES)) - 1} rows)")
        elif c in ("air", "smell", "dry", "wet"):
            if args.mock and c in ("smell", "wet"):
                node.wet = True
            m, g = read(node)
            if args.mock:
                node.wet = False
            refs[{"air": "gas_air", "smell": "gas_smell", "dry": "moisture_dry", "wet": "moisture_wet"}[c]] = g if c in ("air", "smell") else m
            REFS.write_text(json.dumps(refs, indent=1))
            print(f"  {c}: moisture {m:.0f}  gas {g:.0f}")
        elif c == "sample" and len(line) == 3:
            cat, tsp = line[1].lower(), float(line[2])
            lab = int(tsp >= args.contaminated_from)
            if args.mock:
                node.wet = bool(lab)
            m, g = read(node)
            w.writerow([cat, tsp, lab, f"{m:.0f}", f"{g:.0f}", time.strftime("%H:%M:%S")]); f.flush()
            print(f"  saved: {cat} {tsp} tsp -> contaminated={lab}  moisture {m:.0f}  gas {g:.0f}")
        else:
            print("  unknown command")
    f.close()


def fit(args):
    from edgewaste.contamination.oci_model import fit_oci_weights, run_ablation, select_threshold
    from edgewaste.contamination.sensor_normalization import CalibrationAnchors, normalize_gas, normalize_moisture
    if not (SAMPLES.exists() and REFS.exists()):
        raise SystemExit("run 'collect' first: need runs/prototype/oci_samples.csv and oci_references.json")
    refs = json.loads(REFS.read_text())
    need = ["gas_air", "gas_smell", "moisture_dry", "moisture_wet"]
    miss = [k for k in need if k not in refs]
    if miss:
        raise SystemExit(f"missing reference readings: {miss} (use the air, smell, dry and wet commands)")
    r0 = rs_of(refs["gas_air"])
    l_max = -math.log(rs_of(refs["gas_smell"]) / r0)
    if l_max <= 0:
        raise SystemExit("the smell reference gave a lower gas reading than clean air: check the sensor and repeat 'air' and 'smell'")
    anchors = CalibrationAnchors(m_dry=refs["moisture_dry"], m_wet=refs["moisture_wet"], r0=r0, l_min=0.0, l_max=l_max)
    rows = list(csv.DictReader(open(SAMPLES)))
    y = np.array([int(r["contaminated"]) for r in rows])
    if min((y == 0).sum(), (y == 1).sum()) < 8:
        raise SystemExit(f"need at least 8 samples of each class, have {(y == 0).sum()} clean and {(y == 1).sum()} contaminated")
    f_m = np.array([normalize_moisture(float(r["moisture_raw"]), anchors) for r in rows])
    f_g = np.array([normalize_gas(rs_of(float(r["gas_raw"])), anchors) for r in rows])
    model, diag = fit_oci_weights(f_m, f_g, y, use_interaction=True)
    abl = run_ablation(f_m, f_g, y)
    from edgewaste.contamination.oci_model import compute_oci
    sc = {"both": np.array([compute_oci(model, a, b) for a, b in zip(f_m, f_g)]),
          "moisture_only": np.array([compute_oci(model, a, None) for a in f_m]),
          "gas_only": np.array([compute_oci(model, None, b) for b in f_g])}
    thr = {}
    for k, s in sc.items():
        for target in (0.95, 0.90, 0.85):
            try:
                t = select_threshold(y, s, target); t["target"] = target; thr[k] = t; break
            except RuntimeError:
                continue
        else:
            raise SystemExit(f"no usable threshold for {k}: collect more or cleaner samples")
    out = {"anchors": anchors.__dict__, "model": model.__dict__, "thresholds": thr, "diagnostics": diag, "cross_validated_auc": abl, "n_samples": len(y),
           "note": "fitted on real prototype readings; small sample, expect modest separation"}
    CALIB.write_text(json.dumps(out, indent=1))
    print(json.dumps({"n": len(y), "cv_auc": abl, "thresholds": {k: round(v["operating_threshold"], 3) for k, v in thr.items()},
                      "sensitivity": {k: round(v["achieved_sensitivity"], 2) for k, v in thr.items()},
                      "false_positive_rate": {k: round(v["achieved_fpr"], 2) for k, v in thr.items()}}, indent=1))
    print("wrote", CALIB)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT), help="where samples, references and the fitted calibration are kept")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--port", default="COM5"); c.add_argument("--mock", action="store_true"); c.add_argument("--contaminated-from", type=float, default=1.0)
    sub.add_parser("fit")
    a = ap.parse_args()
    OUT = Path(a.out_dir); SAMPLES = OUT / "oci_samples.csv"; REFS = OUT / "oci_references.json"; CALIB = OUT / "oci_calibration.json"
    collect(a) if a.cmd == "collect" else fit(a)
