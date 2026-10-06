# report_appendix.py - appendix a of the project report: sample code, configuration, run outputs and training log
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TITLE = ("APPENDIX A", "SAMPLE CODE, DATA AND SCREEN CAPTURES")


def history_rows():
    h = json.loads((ROOT / "runs/stage1_v2/history.json").read_text())
    return [(str(r["epoch"]), f"{r['train_loss']:.3f}", f"{r['train_acc'] * 100:.1f}%", f"{r['val_loss']:.3f}", f"{r['val_acc'] * 100:.2f}%", f"{r['seconds']:.0f}") for r in h]


def blocks():
    return [
        ("p", "This appendix keeps additional sample code, configuration and run outputs. Only the parts that matter are shown; the full source is in the project repository. "
              "Page numbers are not shown from this appendix onwards."),
        ("h2", "A.1 Attention Fusion and the Hybrid Classifier (Excerpt)"),
        ("p", "The excerpt shows the learned attention over the two backbone streams and the forward pass (Chapter 4, {E:es} to {E:fuse})."),
        ("codeslice", "src/edgewaste/classification/hybrid_model.py", 26, 52),
        ("codeslice", "src/edgewaste/classification/hybrid_model.py", 72, 100),
        ("h2", "A.2 Conformal Calibration and Prediction Sets (Excerpt)"),
        ("p", "The excerpt shows the standard and Beta-corrected ranks and the per-family calibration of Section 4.1.6."),
        ("codeslice", "src/edgewaste/decision_engine/conformal_routing.py", 71, 119),
        ("h2", "A.3 Hazard-Exempt Object-Identity Prior (Excerpt)"),
        ("codeslice", "src/edgewaste/decision_engine/object_identity_prior.py", 93, 119),
        ("h2", "A.4 Training Configuration of the Improved Recipe"),
        ("codeslice", "configs/classifier_convnext_vit.yaml", 16, 51),
        ("h2", "A.5 Training Log of the Improved Recipe"),
        ("p", "The log below is the per-epoch history of the improved recipe on the training platform (training accuracy is measured on augmented images; the first two epochs train only the new layers)."),
        ("tbl", "hist", "Per-Epoch History of the Improved Recipe", ["Epoch", "Train loss", "Train accuracy", "Validation loss", "Validation accuracy", "Seconds"], history_rows(), [1.6, 2.4, 2.8, 2.8, 3.2, 1.8], {"align_cols": ["c"] * 6}),
        ("h2", "A.6 Sample Output of the Video Survey"),
        ("p", "The first rows of the track table written by the video survey (one row per inventory entry) are shown below."),
        ("codefile", "reports/demo/video/tracks.csv", 6),
        ("h2", "A.7 Quick Start and Reproduction Commands"),
        ("code", """pip install -e ".[data,detect,explain,export]"      # CUDA torch first if you have a GPU
python -m edgewaste.taxonomy                         # the 33 classes and their sources
edgewaste-fetch  --config configs/classifier_convnext_vit.yaml
edgewaste-ingest --config configs/classifier_convnext_vit.yaml
edgewaste-split  --config configs/classifier_convnext_vit.yaml
edgewaste-train  --config configs/classifier_convnext_vit.yaml
edgewaste-eval   --config configs/classifier_convnext_vit.yaml --ckpt runs/stage1/best.pt \\
                 --manifest data/splits_colab_reconstructed.csv
python scripts/run_ml_analysis.py                    # reports/ml_analysis
python scripts/run_simulations.py                    # reports/simulations
python scripts/make_demo_snapshots.py                # reports/demo
python scripts/make_demo_gallery.py classes          # reports/demo/gallery (also hazards, gradcam, ...)
edgewaste-video --source clip.mp4 --det-ckpt <detector best.pt> --cls-ckpt runs/stage1/best.pt"""),
        ("h2", "A.8 Additional Screen Captures"),
        ("p", "{F:ax1} shows the routing decisions of Chapter 6 in the form produced by the demonstration script, and {F:ax2} shows the 18-class detector's training summary."),
        ("fig", "ax1", "routing_demo", "Routing Decision Sheet Produced by the Demonstration Script", 13.5),
        ("fig", "ax2", "det_results", "Training Summary of the 18-Class Detector: Losses and Validation Metrics per Epoch", 14.0),
    ]
