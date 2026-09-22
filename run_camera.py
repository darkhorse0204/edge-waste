"""
run_camera.py — one-command launcher for the edge-waste live camera demo.
Each run gets its own session folder under runs/logs/<timestamp>/,
containing both the prediction CSV and that session's Grad-CAM explanations.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

project = Path(__file__).resolve().parent

detector = project / "runs" / "detect" / "taco_single_class" / "weights" / "best.pt"
classifier = project / "runs" / "stage1" / "best.pt"
config = project / "configs" / "stage1.yaml"

session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
session_dir = project / "runs" / "logs" / session_id
log_csv = session_dir / "predictions.csv"
explain_dir = session_dir / "explanations"

for path, label in [(detector, "detector checkpoint"),
                     (classifier, "classifier checkpoint"),
                     (config, "config file")]:
    if not path.exists():
        print(f"ERROR: {label} not found at: {path}")
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

print(f"Launching edge-waste camera pipeline... (session: {session_id})")
result = subprocess.run(cmd)
sys.exit(result.returncode)