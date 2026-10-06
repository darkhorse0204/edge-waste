# report_ch3.py - chapter 3 (technical specification: requirements, data, hardware, software, interfaces)
TITLE = ("CHAPTER 3", "TECHNICAL SPECIFICATION")

FOLDERS = """edge-waste/
  readme.md                       the single reference document (data, model, results, how to run)
  project_status.md               dated log of work and open risks
  configs/                        classifier, Swin variant and detector settings (yaml)
  notebooks/                      Google Colab training notebook
  scripts/                        launchers, demonstrations, analysis and report builders
  reports/ml_analysis/            generated analysis: metrics, tables, figures
  reports/simulations/            simulation results and figures
  reports/demo/                   demonstration figures, annotated video, class tour
  src/edgewaste/
    data/                         download, ingest, split, image loading
    classification/               hybrid model, training, evaluation, inference, ONNX export
    detection/                    YOLO detector, object recogniser, SAM segmentation
    decision_engine/              identity prior, Monte Carlo dropout, conformal routing, rules
    contamination/                sensor scaling, index models, simulated sensors, SHAP
    explainability/               Grad-CAM, LIME
    augmentation/                 DCGAN synthetic images
    federated/                    federated averaging simulation
    analysis/                     metrics, fit diagnostics, calibration, imbalance, baselines
    applications/                 live camera pipeline, video litter survey
  docs/                           planning notes, report (not in version control)"""


def blocks():
    B = []
    B += [
        ("h2", "3.1 System Requirements"),
        ("p", "The requirements were written at the start of the project and refined as results arrived. {T:fr} lists the functional requirements and {T:nfr} the non-functional "
              "requirements, together with how each was verified. The structure follows the practice of ISO/IEC/IEEE 29148 for requirements engineering "
              "(ISO/IEC/IEEE, 2018)."),
        ("tbl", "fr", "Functional Requirements", ["ID", "Requirement", "Verified by"], [
            ("FR1", "Capture an image or video frame and locate each waste item within 500 ms per item.", "Latency measurement (Section 7.12)"),
            ("FR2", "Classify the item into one of 33 classes and report its material family.", "Held-out test set (Section 7.2)"),
            ("FR3", "Cross-check the object name against the material prediction without weakening a hazardous prediction.", "Identity prior and test clip (Section 7.8)"),
            ("FR4", "Estimate uncertainty and send unsure hazardous items to priority review.", "Gate test (Sections 7.8 and 7.9)"),
            ("FR5", "Route by a calibrated set of plausible families with an operator-chosen hazard limit.", "Conformal evaluation (Section 7.8)"),
            ("FR6", "Compute a contamination index from moisture and gas readings, also when one sensor is missing.", "Simulated sensor study (Section 7.10)"),
            ("FR7", "Explain a decision with a heat map and a contamination-score breakdown.", "Demonstration (Chapter 6)"),
            ("FR8", "Produce a de-duplicated inventory from video.", "Video survey (Section 6.7)"),
            ("FR9", "Take part in federated learning rounds without uploading images.", "Federated simulation (Section 7.11)"),
        ], [1.0, 9.0, 4.5], {"font": 9.5}),
        ("tbl", "nfr", "Non-Functional Requirements", ["Area", "Requirement"], [
            ("Performance", "Inference on a laptop below 500 ms per item; the models can be exported to ONNX for a Raspberry Pi 5."),
            ("Reliability", "The contamination module must degrade gracefully when one sensor fails, using the matching single-sensor model."),
            ("Safety", "A hazard prediction must never reach a recycling or compost bin automatically; unsure hazards go to a person."),
            ("Privacy", "No raw image or personal data leaves a unit; only model parameters are shared."),
            ("Reproducibility", "Fixed random seeds, a leakage-checked split, and one script for each reported number."),
            ("Maintainability", "Backbone-agnostic code, one configuration file per experiment, one-line header comment in every source file."),
            ("Portability", "Runs on Windows (development) and Google Colab (training); exports to ONNX."),
        ], [3.0, 11.5], {"font": 9.5}),
        ("h2", "3.2 Dataset Specification"),
        ("p", "The classifier is trained on three public Kaggle datasets, consolidated into one taxonomy. {T:src} lists them. Coarse folders that cannot be mapped to a single "
              "fine-grained class, for example a generic plastic folder, were dropped rather than guessed, because mixing a coarse class with fine ones would make a class depend on "
              "which dataset an image came from and not on the object. One unreadable image in the TrashBox electronic-waste folder was rejected at ingest."),
        ("tbl", "src", "Datasets Used for Classification", ["Source (Kaggle identifier)", "Used for", "Images"], [
            ("Recyclable and Household Waste Classification (alistairking/recyclable-and-household-waste-classification)", "30 item classes; studio and real-world photographs", "15,000"),
            ("Garbage Classification 12 (mostafaabla/garbage-classification)", "battery, clothing, shoes and extra food waste", "added to the classes above"),
            ("TrashBox (minhle13/trashbox)", "electronic waste and medical waste, the hazardous classes no other source has", "added to the classes above"),
            ("Total after consolidation", "33 item classes in 9 material families", "28,203"),
        ], [8.2, 5.2, 2.6]),
        ("fig", "pipe", "data_pipe", "Data Pipeline From the Public Datasets to the Training, Validation and Test Sets", 14.0),
        ("p", "The classifier predicts the item, and the family, which is the physical bin, is derived from it. {F:tax} shows the 33 classes in their nine families with the number of "
              "images of each, and {F:counts} shows the same counts on a logarithmic scale split into training, validation and test images. Twenty-seven classes have about 500 "
              "images each. Six merged classes are larger: clothing (5,825), shoes (2,477), electronic waste (2,406), medical waste (1,565), food waste (1,485) and battery (945). "
              "The ratio of the largest class to the smallest is 11.6."),
        ("fig", "tax", "taxonomy", "The 33 Item Classes in Nine Material Families", 14.4),
        ("fig", "counts", "counts", "Number of Images per Class, Split Into Training, Validation and Test Sets (Logarithmic Scale)", 14.4),
        ("p", "A hierarchy is used because a sorting facility acts on the family. Family accuracy is therefore the operational metric, while errors inside one family, such as "
              "aluminium against steel food cans, are recognition errors with no routing consequence. Hazardous items are a family of their own so that their recall is never "
              "averaged away by the larger classes."),
        ("h3", "3.2.1 Splits and Leakage Prevention"),
        ("p", "The data are split per class in the proportion 70 : 15 : 15 with a fixed seed of 42 ({T:split}). The validation set is used for early stopping and for every choice of "
              "hyperparameter or threshold; the test set is scored once per model. Each processed image is named by a hash of its path relative to the source root, so that "
              "the same photograph can never appear twice."),
        ("tbl", "split", "Split Sizes", ["Split", "Images", "Use"], [
            ("Training", "19,739", "learning the weights (with augmentation)"),
            ("Validation", "4,232", "early stopping, learning-rate and threshold choices, temperature fitting"),
            ("Test", "4,232", "final scores, scored once per model; 738 hazardous images"),
        ], [3.0, 3.0, 8.5], {"align_cols": ["l", "c", "l"]}),
        ("p", "Two data problems were found and fixed during the project, and both were caught because a result looked too good. First, the file-name hash was originally taken over an "
              "operating-system-specific path string, so Windows and Colab produced different names, different orders and therefore different splits. Re-scoring the Colab-trained "
              "checkpoint on a locally rebuilt split gave 94.52%, because about 70% of its “test” images had been training images. The hash now uses the POSIX path relative to the "
              "source root, and the exact Colab split was recovered and verified: the checkpoint scores exactly 3,779 of 4,232 (89.30%) on it. Second, a Colab run showed 1,234 steps "
              "per epoch instead of 617, because stale copies of every image were left on the drive from an earlier run; the split now ignores files that are not listed in the latest "
              "provenance record, and the training notebook stops if the split size differs from the provenance size."),
        ("h3", "3.2.2 Preprocessing and Augmentation"),
        ("p", "Images are resized to 224 × 224 pixels and normalised with the ImageNet mean and standard deviation, which matches the pretraining of the backbones. Training images "
              "are also augmented to imitate camera pose, lighting and occlusion on a conveyor ({T:aug}). Validation and test images are not augmented."),
        ("tbl", "aug", "Training-Only Augmentation", ["Step", "Setting"], [
            ("Resize, then random resized crop", "resize to 256, crop 224, scale 0.7 to 1.0"),
            ("Horizontal flip", "probability 0.5"), ("Rotation", "up to ±20 degrees"),
            ("Colour jitter", "brightness, contrast and saturation 0.2"), ("Gaussian blur", "probability 0.2"), ("Random erasing", "probability 0.1"),
        ], [6.0, 8.5]),
        ("p", "For the object detector, the TACO litter dataset (Proença & Simões, 2020) was prepared in two forms: a single class “waste item” for localisation and an 18-class "
              "form that names the object (bottle, can, cup, cigarette, plastic bag or wrapper, and others). The validation split has 900 images."),
        ("h2", "3.3 Hardware Specification"),
        ("p", "Development and training used the machines in {T:dev}. The training of the first model and the improved recipe ran on Google Colab, because one epoch takes about 55 "
              "minutes on the laptop GPU and about 4 minutes on the Colab GPU. The planned prototype is described in {T:proto}; its architecture is shown in {F:hw}."),
        ("tbl", "dev", "Development and Training Hardware", ["Machine", "Specification", "Used for"], [
            ("Laptop (Windows 11)", "NVIDIA RTX 2050 GPU with 4 GB memory, CUDA 12.1", "detector training, analysis, simulations, demonstrations, latency tests"),
            ("Google Colab", "NVIDIA T4 GPU (about 4 minutes per epoch)", "classifier training (first model and improved recipe)"),
        ], [3.2, 6.0, 5.3]),
        ("tbl", "proto", "Planned Prototype Hardware (Not Yet Built)", ["Part", "Planned choice", "Role"], [
            ("Edge computer", "Raspberry Pi 5 class single-board computer", "runs the ONNX detector, classifier and decision engine"),
            ("Camera", "Camera module or USB camera", "image or video of the item"),
            ("Moisture sensor", "Capacitive moisture sensor", "wetness score f<sub>m</sub>"),
            ("Gas sensor", "MQ-135 type metal-oxide sensor", "gas score f<sub>g</sub>"),
            ("Converter", "16-bit analogue-to-digital converter", "reads the two sensors"),
            ("Actuation", "Servo diverter, motor driver and a small conveyor belt", "moves the item to the chosen bin"),
            ("Bins", "Eight material bins and one hazardous bin", "physical destinations"),
        ], [3.0, 5.5, 6.0]),
        ("fig", "hw", "hw", "Architecture of the Planned Hardware Prototype", 14.4),
        ("h2", "3.4 Software Specification"),
        ("p", "The system is written in Python and packaged as the installable package edgewaste with twelve console commands. {T:sw} lists the main libraries and the versions used."),
        ("tbl", "sw", "Software Environment", ["Component", "Version or choice", "Purpose"], [
            ("Python", "3.11", "language"), ("PyTorch / torchvision", "2.5.1 (CUDA 12.1) / 0.20.1", "deep learning"),
            ("timm", "1.0.29", "pretrained ConvNeXt, Vision Transformer and Swin backbones"),
            ("Ultralytics", "8.4.157", "YOLO26n detector, MobileSAM"),
            ("scikit-learn", "1.9.0", "PCA, LDA, SVM, kNN, random forest, logistic regression, metrics"),
            ("NumPy / pandas / matplotlib", "2.4.6 / 3.0.5 / 3.11.2", "numerics, tables, figures"),
            ("OpenCV", "4.14", "video and image handling"), ("ONNX / ONNX Runtime", "1.23.0 / 1.30.0", "model export and checking"),
            ("grad-cam / lime / shap", "1.5.7 / 0.2.0.1 / 0.51.0", "explanations"),
            ("Mermaid, python-docx", "command-line renderer, 1.2", "diagrams and this report"),
        ], [4.6, 4.5, 5.4]),
        ("h2", "3.5 Performance Specification"),
        ("p", "{T:spec} states the performance targets set for the software stage. They are compared with the measured values in Chapter 7."),
        ("tbl", "spec", "Performance Targets", ["Quantity", "Target", "Notes"], [
            ("Item accuracy (33 classes)", "at least 90%", "first model 89.30%; improved recipe 93.83%"),
            ("Family (routing) accuracy", "at least 94%", "first model 94.14%; improved recipe 97.28%"),
            ("Hazard recall", "at least 95%", "first model 93.50%; improved recipe 98.37%"),
            ("Hazard leakage with the conformal gate", "below the chosen α<sub>H</sub>", "tested for α<sub>H</sub> from 0.05 to 0.005"),
            ("Latency per item (laptop)", "below 500 ms", "14 ms on the GPU, 116 ms on the CPU"),
            ("Contamination sensitivity", "at least 95% in each case", "simulated sensors"),
        ], [5.0, 3.5, 6.0]),
        ("h2", "3.6 Repository and Interfaces"),
        ("p", "The whole project is kept in one repository. The folder layout below shows where each part lives. A single readme file is the reference document and is updated with "
              "every change. Every source, configuration and document file starts with a one-line lowercase comment that states what it is."),
        ("code", FOLDERS),
        ("p", "{T:cmd} lists the main commands. Each can be run on its own and writes its output under the reports or runs folders."),
        ("tbl", "cmd", "Main Commands", ["Command", "What it does"], [
            ("edgewaste-fetch / edgewaste-ingest / edgewaste-split", "download, consolidate and split the datasets"),
            ("edgewaste-train", "train the hybrid classifier (or run the Colab notebook)"),
            ("edgewaste-eval", "evaluate a checkpoint on the test split, refusing a split that does not reproduce the recorded validation accuracy"),
            ("scripts/run_ml_analysis.py", "metrics, fit diagnostics, calibration, imbalance, classical baselines, projections, model cost"),
            ("scripts/run_simulations.py", "uncertainty, conformal, routing, damaged-image and sensor-loss simulations"),
            ("scripts/make_demo_snapshots.py, make_demo_gallery.py", "demonstration figures and videos (Chapter 6)"),
            ("edgewaste-video", "video litter survey with tracking and an inventory report"),
            ("python scripts/run_live_camera.py", "live detect, classify and route demonstration"),
        ], [6.2, 8.3]),
        ("h2", "3.7 Use Cases and Software Structure"),
        ("p", "{F:uc} shows who uses the system and for what. The operator reviews items flagged for manual review and inspects explanations, the facility manager sets the hazard "
              "limit and the review budget and runs litter surveys, and the maintainer recalibrates the thresholds for a new site and joins federated rounds. {F:cls} shows the main "
              "software modules and how they depend on each other."),
        ("fig", "uc", "uc", "Use Cases of the Waste Sorting Unit", 9.0, 15.0),
        ("fig", "cls", "cls_d", "Main Software Modules and Their Relations", 13.0),
    ]
    return B
