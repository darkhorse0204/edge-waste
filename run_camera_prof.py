"""
run_camera_prof.py
Launch the live camera demo using the NEW classifier
(runs/stage1_prof_dataset/best.pt).
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

project = Path(__file__).resolve().parent

# YOLO detector
detector = project / "runs" / "detect" / "taco_single_class" / "weights" / "best.pt"

# NEW classifier
classifier = project / "runs" / "stage1_prof_dataset" / "best.pt"

config = project / "configs" / "stage1_prof_dataset.yaml"

session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
session_dir = project / "runs" / "logs_prof" / session_id

session_dir.mkdir(parents=True, exist_ok=True)

log_csv = session_dir / "predictions.csv"
explain_dir = session_dir / "explanations"

for path, label in [
    (detector, "Detector checkpoint"),
    (classifier, "Classifier checkpoint"),
    (config, "Config file"),
]:
    if not path.exists():
        print(f"ERROR: {label} not found:")
        print(path)
        sys.exit(1)

cmd = [
    sys.executable,
    "-m",
    "edgewaste.pipeline",
    "--config",
    str(config),
    "--det-ckpt",
    str(detector),
    "--cls-ckpt",
    str(classifier),
    "--camera",
    "--log-csv",
    str(log_csv),
    "--explain-dir",
    str(explain_dir),
]

print("=" * 70)
print("Launching EdgeWaste Camera")
print("Detector :", detector)
print("Classifier :", classifier)
print("Logs :", session_dir)
print("=" * 70)

subprocess.run(cmd)