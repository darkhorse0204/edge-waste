# report_ch2.py - chapter 2 (project description and goals: problem, goals, scope, literature and patent survey, research gap)
TITLE = ("CHAPTER 2", "PROJECT DESCRIPTION AND GOALS")

LIT_ROWS = [
    ("1", "Alkılınç et al. (2025). Deep ensemble learning model for waste classification systems.",
     "Combines 16 pretrained CNNs by averaging and weighted averaging; assessed with Grad-CAM.",
     "Small, controlled datasets; the ensemble is costly to compute, which limits real-time edge use."),
    ("2", "Verber et al. (2026). Image-based waste classification using a hybrid deep learning architecture with transfer learning and edge AI deployment.",
     "Hybrid attention-fusion model of ResNet, EfficientNet and MobileNetV3; deployed on a Jetson Nano.",
     "Modest dataset; RGB images in controlled conditions; occluded or contaminated samples under-represented."),
    ("3", "Partosan et al. (2026). Campus-scale real-time waste classification on Raspberry Pi 5 using You Only Look Once version 8.",
     "YOLOv8 detector with strong augmentation, running in real time on a Raspberry Pi 5.",
     "Needs quantisation or pruning for faster embedded inference and wider scale."),
    ("4", "Alnanih et al. (2025). Advancing sustainability through an IoT-driven smart waste management system with software engineering integration.",
     "Regional IoT design with ultrasonic and flame sensors and a cloud database for bin monitoring.",
     "No material segregation; accuracy testing of the ultrasonic sensor used a limited sample."),
    ("5", "Nahiduzzaman et al. (2025). An automated waste classification system using deep learning techniques.",
     "Three-stage classification with a lightweight network (1.09 million parameters).",
     "Accuracy falls as the number of sub-classes grows (85.25% in the final stage)."),
    ("6", "Arun (2025). Investigation of a deep learning-based waste recovery framework for sustainability and a clean environment using IoT.",
     "Joins IoT sensors for real-time data collection with deep learning for automatic identification.",
     "Needs better algorithms and sensors before it extends reliably to hazardous and medical waste."),
    ("7", "Alsabt et al. (2024). Optimizing waste management strategies through artificial intelligence and machine learning.",
     "Compares SVM, random forest and XGBoost with optimisation on a large World Bank dataset.",
     "Data quality and completeness problems; standard waste-management data are needed."),
    ("8", "Hussain et al. (2024). A comprehensive review on deep learning-based data fusion.",
     "Thorough review of early, intermediate, late and hybrid fusion strategies.",
     "Aligning and synchronising sensors with different sampling rates remains difficult."),
    ("9", "Casao et al. (2024). SpectralWaste dataset: Multimodal data for waste sorting automation.",
     "Hyperspectral and RGB data from a real sorting plant, with several segmentation models.",
     "Some classes segment poorly with current architectures; semi-supervised refinement is needed."),
    ("10", "Raghavendra et al. (2026). Multimodal sensor fusion for waste management using graph neural networks.",
     "Models spatial and temporal relations across sensor locations under realistic constraints.",
     "Only 500 samples, which limits transfer of knowledge from richer modalities."),
    ("11", "Dipo et al. (2025). Real-time waste detection and classification using a YOLOv12-based deep learning model.",
     "Automatic waste detection and classification in real time.",
     "Limited dataset scope; extreme weather affects sensitivity; edge processing is demanding."),
    ("12", "Chahine and Ghazal (2017). Automatic sorting of solid wastes using sensor fusion.",
     "PLC-based sorter using several sensors to tell wood, glass, plastic and metal apart.",
     "Depends on careful range adjustment of the capacitive proximity sensor."),
    ("13", "He et al. (2026). A survey on uncertainty quantification methods for deep learning.",
     "A taxonomy of uncertainty methods by data and model uncertainty.",
     "Out-of-distribution robustness is still weak in the surveyed methods."),
    ("14", "Radchenko and Fill (2024). Uncertainty estimation in multi-agent distributed learning for AI-enabled edge devices.",
     "Bayesian neural networks for distributed processing and uncertainty on edge devices.",
     "Assumes every message is received; ignores agent failure and complete network breakdown."),
    ("15", "Alatawi (2025). Edge computing and federated learning for privacy-preserving IoT analytics.",
     "Hierarchical federated learning with adaptive learning rates and differential privacy.",
     "Relies on partial central coordination; adversarial attacks are not fully handled."),
    ("16", "Chu et al. (2018). Multilayer hybrid deep-learning method for waste classification and recycling.",
     "Camera plus sensors in public areas; over 90% accuracy in two test scenarios and better than an image-only network.",
     "Sensor values are extra inputs of one classifier; no calibrated contamination score, uncertainty gate or hazard rule is described."),
    ("17", "White et al. (2020). WasteNet: Waste classification at the edge for smart bins.",
     "A convolutional network for low-power devices with 97% accuracy on six categories.",
     "Six broad categories and vision only; no sensors or uncertainty."),
    ("18", "Yang and Thung (2016). Classification of trash for recyclability status.",
     "The widely used TrashNet photographs of six broad categories.",
     "Few images with plain backgrounds; broad categories only."),
    ("19", "Proença and Simões (2020). TACO: Trash annotations in context for litter detection.",
     "Public photographs of litter in the wild with bounding boxes.",
     "Labels name objects rather than materials and the set is small."),
]

PAT_ROWS = [
    ("US 10,799,915 B2", "AMP Robotics Corp.; priority 28 July 2017; granted 13 October 2020",
     "Imaging sensors and a neural network identify recyclable items on a conveyor; pushers divert target items.",
     "Vision classifier with automatic diversion; no object–material cross-check, hazard rule or contamination index."),
    ("EP 4 055 520 B1", "Tomra Sorting GmbH; priority 4 November 2019; granted 1 January 2025",
     "A convolutional network analyses multispectral or hyperspectral data to detect, classify and segment objects.",
     "Spectral sensing in an industrial sorter; this project uses low-cost moisture and gas sensors and a calibrated index."),
    ("US 11,969,764 B2", "Battelle Energy Alliance LLC and Sortera Technologies Inc.; priority 18 July 2016; granted 30 April 2024",
     "Fusion of several sensors and machine learning sorts plastics by polymer type and composition.",
     "Sensor fusion for composition; no hazard-dominant, uncertainty-aware routing."),
    ("US 11,875,301 B2", "Heil Co.; priority 27 July 2018; granted 16 January 2024",
     "Refuse contamination analysis: cameras on collection vehicles and machine learning find contamination in emptied containers.",
     "Vision only and for vehicles; this project senses contamination per item with physical sensors."),
    ("US 11,335,086 B2", "Fidelity AG Inc.; priority 21 March 2020; granted 17 May 2022",
     "Automated waste management: image classification into recyclable, trash or compost with feedback to the user.",
     "Three broad groups and one model; this project has 33 classes, uncertainty and hazard handling."),
    ("WO 2024/207048 A1", "Commonwealth Scientific and Industrial Research Organisation; priority 5 April 2023; published 10 October 2024",
     "Waste sorting method and apparatus: camera, metal detector, weight and ultrasonic sensors with neural networks and a tilting chute.",
     "Closest multi-sensor reference; no cross-check with hazard exemption, family-set routing or contamination index."),
    ("US 11,104,512 B2", "CleanRobotics Technologies Inc.; priority 15 July 2016; granted 31 August 2021",
     "Automatic sorting of waste: several sensors decide recyclable or not, and the item is sent to the right bin.",
     "Two-way decision; no fine-grained classes or hazard limit."),
    ("US 11,961,054 B1", "Prince Mohammad Bin Fahd University; granted 16 April 2024",
     "Waste management system: image sorting with motors plus separate IoT sensors for bin level, temperature, humidity and gas.",
     "The sensors monitor the bin; this project scores contamination for each item."),
    ("AU 2021101744 A4", "Innovation patent; granted 3 June 2021",
     "Waste segregation with machine learning: camera, ultrasonic and wet/dry sensors, servo flaps and five broad bins.",
     "Threshold-based wet/dry sensing; this project fits a statistical model with a sensitivity target."),
]

GAP_ROWS = [
    ("More than 15 fine-grained classes", "Rare; mostly 3 to 12 broad classes", "Not the aim", "33 classes in 9 families"),
    ("Recognition and routing accuracy reported separately", "Not reported", "Not reported", "Yes (item, family, hazard recall)"),
    ("Object–material cross-check that never weakens a hazard", "Not reported", "Not reported", "Yes (hazard-exempt prior)"),
    ("Calibrated uncertainty and two-tier hazard gate", "Rare", "Rare", "Yes (Monte Carlo dropout, temperature scaling)"),
    ("Stated limit on hazards reaching recycling", "Not reported", "Not reported", "Yes (conformal family sets)"),
    ("Contamination from low-cost sensors that survives sensor loss", "No sensors", "Sensors as extra inputs", "Three-model Organic Contamination Index"),
    ("Explanations of decisions", "Sometimes Grad-CAM", "Rare", "Grad-CAM, LIME, SHAP, SAM outline"),
    ("Privacy-preserving learning across units", "Rare", "Rare", "Federated averaging (simulated)"),
    ("One inventory entry per item in video", "No", "No", "Yes (ByteTrack-based)"),
]


def blocks():
    B = []
    B += [
        ("h2", "2.1 Project Description"),
        ("p", "CAPS-FCL is a waste-sorting pipeline for a small unit that sits at the point where waste is thrown away, such as a smart bin or a short conveyor. An item enters the "
              "view of a camera. An object detector finds it and names the object. A material classifier looks at the item and gives a probability for each of 33 item classes, "
              "which belong to nine material families: plastic, paper, cardboard, glass, metal, organic, styrofoam, textile and hazardous. The object name and the material "
              "prediction are cross-checked, and the system measures how sure it is. Two low-cost sensors, one for moisture and one for gas given off by decaying matter, "
              "give a contamination score. A decision engine then chooses one of four outcomes: the hazardous bin, priority manual review, a contamination reject, or the bin of "
              "the identified family. Many units can improve a shared model by sending parameters, not images, to an aggregation server."),
        ("p", "The project follows the staged plan of its blueprint. Stage 1, the core vision pipeline, and the software parts of Stages 2 and 3 (contamination index, "
              "uncertainty, decision engine, explainability and federated learning) are built and evaluated on real held-out images. The physical sensors, the conveyor "
              "and the Raspberry Pi deployment are the next stage, and wherever sensor data appear in this report they are simulated and marked as such."),
        ("h2", "2.2 Problem Definition"),
        ("p", "Existing automated sorters depend mainly on single-modal visual classification with one network. They achieve high accuracy in controlled conditions, but they "
              "struggle with contaminated waste, visually similar materials and changing conditions, and they are hard to deploy on small devices. Collecting images centrally to "
              "improve them raises privacy and bandwidth concerns. The problem can be stated formally as follows."),
        ("p", "Given an image x of one waste item and, optionally, a moisture reading and a gas reading, the system must output (a) the item class y among C = 33 classes and its "
              "material family f(y) among nine families, (b) a measure of how sure it is, (c) a contamination score, and (d) one action: send the item to a bin, to the hazardous "
              "bin, to priority review or to a contamination reject. The design must meet four requirements at the same time: high routing accuracy, a stated and checkable limit on the "
              "share of hazardous items that reach a recycling bin, a contamination decision that still works when one sensor fails, and a latency of less than 500 ms per item on "
              "an edge-class computer."),
        ("h2", "2.3 Goals and Measurable Targets"),
        ("p", "Each objective of Section 1.1 was turned into a measurable target, listed in {T:goals} with the status at the time of writing. The results are given in Chapter 7."),
        ("tbl", "goals", "Goals, Measurable Targets and Status", ["No.", "Goal", "Measurable target", "Status"], [
            ("1", "Fine-grained recognition with a separate routing check", "At least 15 classes; item, family and hazard-recall metrics reported separately", "Done: 33 classes, 89.30% / 94.14% / 93.50% (first model); 93.83% / 97.28% / 98.37% (improved recipe)"),
            ("2", "Hazard safety with a stated limit", "Hazard leakage below the operator's limit α<sub>H</sub> for items like the calibration set", "Done in software: leakage 6.5% falls to 1.57% at 7.8% review"),
            ("3", "Calibrated uncertainty", "Expected calibration error reduced without changing predictions", "Done: 16.6% falls to 7.3%"),
            ("4", "Contamination detection that survives sensor loss", "Sensitivity of at least 95% in every sensor-availability case", "Done on simulated sensors (94.9% and 94.5% with one channel lost); physical sensors pending"),
            ("5", "Explanations", "Grad-CAM, LIME and SHAP available for every decision type", "Done"),
            ("6", "Privacy-preserving learning", "Shared model reaches the accuracy of a central model without sharing images", "Done in simulation: 89.4% against 89.0% central"),
            ("7", "Video surveys", "One inventory entry per tracked item", "Done: 160 detections merged into 16 entries on a test clip"),
            ("8", "Edge readiness", "Exported model equals the original; latency below 500 ms", "Done on a laptop (14 ms GPU, 116 ms CPU); Raspberry Pi pending"),
        ], [0.8, 4.2, 4.8, 5.2]),
        ("h2", "2.4 Scope and Assumptions"),
        ("p", "The scope covers data engineering, model design and training, the decision logic, calibration and uncertainty, explanations, federated learning in simulation, "
              "ONNX export and the video survey. The following assumptions apply."),
        ("bul", [
            "Items are presented one at a time to the classifier, as crops from the detector, in light that is similar to the training photographs.",
            "Calibration data for the conformal thresholds come from the same kind of items as the data the unit will see; the guarantee is stated for such items.",
            "Sensor data are simulated until physical sensors are built; the simulated generator follows the planned calibration procedure (two item categories, five residue levels).",
            "Training uses public datasets only. Images of the unit's own site are expected to be added later and the thresholds recalibrated.",
        ]),
        ("h2", "2.5 Literature Survey"),
        ("p", "More than fifteen recent papers were studied. They fall into five groups. The studies are summarised in {T:lit} with their merits and demerits."),
        ("h3", "2.5.1 Vision-Based Waste Classification"),
        ("p", "Most work classifies waste from one camera image. Early studies fine-tuned a convolutional network on the TrashNet photographs of six categories (Yang & Thung, 2016; "
              "Bircanoğlu et al., 2018) or compared network designs (Ruiz et al., 2019). Recent studies use ensembles of many networks (Alkılınç et al., 2025), hybrids of lightweight "
              "networks with transfer learning (Verber et al., 2026), multi-stage classifiers (Nahiduzzaman et al., 2025) and real-time detectors of the YOLO family "
              "(Dipo et al., 2025; Partosan et al., 2026). They reach high accuracy, but the datasets are small and controlled, the number of categories is small, and accuracy falls as "
              "the categories become finer."),
        ("h3", "2.5.2 Sensors, Internet of Things and Fusion"),
        ("p", "Sensor-based systems range from a programmable-controller sorter that tells wood, glass, plastic and metal apart with proximity sensors (Chahine & Ghazal, 2017) to IoT "
              "bins that monitor fill level and flames (Alnanih et al., 2025) and frameworks that join sensors with deep learning (Arun, 2025). A survey of deep-learning data fusion "
              "describes early, intermediate, late and hybrid fusion and warns about synchronising sensors with different sampling rates (Hussain et al., 2024). Multimodal data sets "
              "(Casao et al., 2024) and graph-network fusion (Raghavendra et al., 2026) point the same way. The closest published work to the sensor part of this project is "
              "the multilayer hybrid system of Chu et al. (2018), which adds sensor readings to the camera image as extra inputs of one classifier; this project instead computes a "
              "calibrated contamination index."),
        ("h3", "2.5.3 Uncertainty and Reliability"),
        ("p", "Uncertainty quantification is surveyed by He et al. (2026), and Bayesian approaches for edge devices are explored by Radchenko and Fill (2024). Foundational methods "
              "(Monte Carlo dropout, deep ensembles, temperature scaling, selective classification and conformal prediction) are summarised in Section 1.3. None of the waste-sorting "
              "studies above gives the operator a stated limit on how often a hazardous item may reach recycling."),
        ("h3", "2.5.4 Federated and Edge Learning"),
        ("p", "Federated learning for privacy-preserving analytics is studied by Alatawi (2025) and by McMahan et al. (2017) in general. Most smart-bin "
              "systems work in isolation, without the shared learning that city-scale deployment would need."),
        ("h3", "2.5.5 Datasets and Detection"),
        ("p", "TACO provides photographs of litter in the wild with bounding boxes (Proença & Simões, 2020), and SpectralWaste provides multimodal data from a sorting plant "
              "(Casao et al., 2024). Kaggle datasets of household waste photographs supply the material classes used in this project (Chapter 3)."),
        ("tbl", "lit", "Summary of the Literature Survey", ["S.No.", "Study", "Merits", "Demerits"], LIT_ROWS, [1.3, 4.9, 4.3, 4.6], {"font": 9}),
        ("h2", "2.6 Patent Survey"),
        ("p", "A search of Google Patents found nine granted or published patents that are close to the system in different ways: vision-based diversion, spectral sensing, "
              "sensor fusion, contamination analysis, broad-category image classification and multi-sensor sorters. They are listed in {T:pat} with the main difference from this project. "
              "The reading is a first pass by the team and compares only the main idea of each patent."),
        ("tbl", "pat", "Related Patents and Their Difference From This Project", ["Patent", "Owner and dates", "What it describes", "Difference"], PAT_ROWS, [2.4, 3.8, 4.6, 4.4], {"font": 9}),
        ("h2", "2.7 Findings and Research Gap"),
        ("p", "The survey leads to five findings. First, most work is single-modal vision with few broad categories, and accuracy is reported as one number. Second, the multimodal "
              "work rarely combines the camera with chemical or moisture sensors to detect organic residue, and when it does the sensors are extra classifier inputs. Third, the "
              "best ensembles are too heavy for small devices. Fourth, real-time explanation and uncertainty on the edge are rare, and no work gives a stated hazard limit. Fifth, "
              "smart-bin systems rarely learn collectively while keeping images private. {T:gap} compares the typical situation with this project."),
        ("tbl", "gap", "Research Gap: Surveyed Work and This Project Compared", ["Capability", "Vision-only sorters (surveyed)", "Camera plus sensor hybrids (surveyed)", "This project"], GAP_ROWS, [4.6, 3.4, 3.4, 3.8], {"font": 9}),
        ("h2", "2.8 Contributions of the Project"),
        ("p", "The contributions of the project are summarised below. Four of them (the first four) are mechanisms that work together in the decision logic, and the rest are supporting results."),
        ("num", [
            "A hazard-exempt object–material cross-check: object identity can narrow a non-hazardous material guess, but a hazardous prediction is never weakened by it.",
            "A two-tier hazard gate based on calibrated uncertainty: a confident hazard goes to the hazardous bin and an unsure hazard goes to priority review, never to recycling.",
            "Calibrated routing by sets of material families (conformal prediction), which gives the operator a stated limit α<sub>H</sub> on how often a hazard may reach a recycling bin.",
            "A three-model Organic Contamination Index that switches to a dedicated single-channel model when a sensor fails and keeps its sensitivity target in every case.",
            "A 33-class, 9-family taxonomy with separate item, family and hazard-recall reporting, and a leakage-checked data split.",
            "A full machine-learning analysis (fit diagnostics, calibration, class imbalance, PCA, LDA, SVM and other classical baselines) and a diagnosis of fine-tuning distortion that led to an improved recipe.",
            "A simulation suite that tests each mechanism, including damaged-image robustness, and reports unfavourable results as found.",
        ]),
    ]
    return B
