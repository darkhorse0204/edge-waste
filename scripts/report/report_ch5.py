# report_ch5.py - chapter 5 (schedule, tasks and milestones, taken from the repository history)
TITLE = ("CHAPTER 5", "SCHEDULE, TASKS AND MILESTONES")


def blocks():
    B = [
        ("p", "The project ran from the start of the semester on 6 July 2026 and follows the staged plan of the project blueprint, which deliberately does not build all ten modules "
              "at once so that there is always a working system. Stage 1 is the core vision pipeline, Stage 2 adds sensors, the contamination index, uncertainty and the decision engine, "
              "and Stage 3 adds the research features: federated learning, explainability, optimisation and the full evaluation. The dates below are taken from the history of the "
              "project repository and from the review calendar."),
        ("h2", "5.1 Project Timeline"),
        ("p", "{F:gantt} shows the timeline. Work was dense in late September, when faculty feedback asked for far more than 15 classes and the taxonomy and the models were rebuilt "
              "in a few days, and the analysis and simulation suites were added. The hardware prototype is planned after the final review and is drawn hatched."),
        ("fig", "gantt", "gantt", "Project Timeline From the Start of the Semester to the Final Review and the Planned Prototype", 14.4),
        ("h2", "5.2 Milestones"),
        ("tbl", "miles", "Milestones and Their Dates", ["Date", "Milestone", "Evidence"], [
            ("06 Jul 2026", "Start of the project period", "semester start; certificate period"),
            ("19 Jul 2026", "Stage 1 pipeline: ConvNeXt and Vision Transformer classifier", "repository history"),
            ("20 Jul 2026", "Decision on a public-data taxonomy of 7 classes", "planning notes"),
            ("30–31 Jul 2026", "YOLO detect-then-classify stage, handover notes and training notebook", "repository history"),
            ("03 Aug 2026", "Live inference stabilised, object-aware explanation, project audit report", "repository history, audit report"),
            ("19 Aug 2026", "Review 1 submitted with literature survey of 15 papers", "review submission"),
            ("22 Sep 2026", "Stage 2 and 3 in software; faculty asks for more than 15 classes", "repository history"),
            ("23 Sep 2026", "33-class taxonomy; 18-class detector; video survey; classifier trained on Colab (89.30%); leakage bug found and fixed; ONNX export", "repository history, training log"),
            ("24 Sep 2026", "Grad-CAM, LIME, SHAP, SAM, DCGAN and Swin comparison added", "repository history"),
            ("25 Sep 2026", "Conformal hazard-leakage routing", "repository history"),
            ("30 Sep 2026", "Repository restructured; machine-learning analysis suite; fine-tuning diagnosis", "repository history"),
            ("01 Oct 2026", "Improved recipe retrained (93.83%); simulation suite; demonstration gallery", "training log, simulation reports"),
            ("05–09 Oct 2026", "Pre-final (guide) review", "review calendar"),
            ("21 Oct 2026", "Final review and report upload", "review calendar"),
        ], [2.6, 8.2, 3.7], {"font": 9.5}),
        ("h2", "5.3 Tasks by Module"),
        ("p", "{T:tasks} lists the tasks of each module, who benefits from them and their status. “Software” means built and tested on real held-out data or on simulated sensors; "
              "“planned” means that it needs physical hardware."),
        ("tbl", "tasks", "Tasks and Status by Module", ["Module", "Main tasks", "Status"], [
            ("Data engineering", "download, map to taxonomy, hash-named images, stratified split, leakage checks", "software, done"),
            ("Vision model", "hybrid classifier, attention fusion, training recipe, improved recipe, ONNX export", "software, done"),
            ("Detector and identity prior", "TACO preparation, 1-class and 18-class YOLO, COCO recogniser, object–material prior", "software, done"),
            ("Uncertainty and routing", "Monte Carlo dropout, temperature scaling, two-tier gate, conformal family sets", "software, done"),
            ("Contamination index", "scaling, three models, threshold, SHAP, simulated data generator", "software, done; sensors planned"),
            ("Explainability and segmentation", "Grad-CAM, LIME, SHAP, MobileSAM", "software, done"),
            ("Federated learning and augmentation", "federated averaging, DCGAN", "software, done in simulation"),
            ("Video survey", "tracking, voting, inventory report", "software, done"),
            ("Analysis and simulation", "metrics, fit diagnostics, calibration, imbalance, baselines, damaged-image and sensor-loss studies", "software, done"),
            ("Hardware prototype", "Raspberry Pi deployment, sensors, conveyor, servo diverter, field calibration", "planned"),
        ], [3.6, 7.6, 3.3], {"font": 9.5}),
        ("h2", "5.4 Risks and Their Handling"),
        ("tbl", "risk", "Risk Register", ["Risk", "Effect", "Handling", "Status"], [
            ("Data leakage between training and test sets", "inflated accuracy", "path-based hashing, validation-accuracy guard in the evaluation command, split recovery", "fixed"),
            ("Domain shift to field footage", "accuracy falls, hazard risk rises", "uncertainty gate, conformal sets, plan to add site images and recalibrate", "open"),
            ("Sensors not yet available", "contamination index untested on real data", "simulated generator that follows the calibration procedure; results labelled simulated", "open"),
            ("Limited GPU memory and time", "slow training", "Colab for training, small backbones, one job at a time", "managed"),
            ("Ambiguous look-alike classes", "lower item accuracy", "family-level routing so that these errors do not change the bin", "managed"),
            ("Improved recipe not reproduced locally", "reported figure rests on the training platform", "checkpoint guard added; the Colab split files are needed to re-score", "open"),
        ], [3.6, 3.0, 5.7, 2.2], {"font": 9.5}),
        ("h2", "5.5 Plan After the Final Review"),
        ("p", "The next stage, Project II, builds the physical unit. The planned steps are: (1) buy and assemble the sensors, the single-board computer, the conveyor and the diverter; "
              "(2) run the calibration protocol with real residues at five levels on two item categories, fit the three contamination models and verify the sensitivity target; "
              "(3) export the models to the board and measure the real latency; (4) collect images at the site, recalibrate the conformal thresholds and test the hazard limit on field data; "
              "(5) run a federated round across at least two physical units; and (6) re-score the improved model on the exact training-platform split to confirm the reported figures."),
    ]
    return B
