# report_front.py - text of the front matter: title, team, executive summary, abbreviations and symbols
"""Front-matter content for the BITE497J Project I report."""

TITLE = "Edge-Deployed Waste Classification with Sensor-Augmented Vision"
STUDENTS = [("Ansh Jerath", "23BIT0123"), ("Arnav Mishra", "23BIT0142"), ("Bhavesh Singh Thakur", "23BIT0199")]
GUIDE = "Dr. Valarmathi B."
GUIDE_FULL = "Dr. Valarmathi B., Professor Grade 1"
DOMAIN = "Artificial Intelligence, the Internet of Things, Waste Management and Federated Learning"
DATE = "November, 2026"

EXEC_SUMMARY = (
    "Automated waste sorters usually rely on one camera and one classifier. They give a single answer per item, say nothing about how sure they are, "
    "cannot see contamination such as food residue, and can send hazardous items such as batteries to recycling. This project, Edge-Deployed Waste "
    "Classification with Sensor-Augmented Vision (CAPS-FCL), builds an edge-ready sorting pipeline in software. A YOLO detector finds each item. "
    "A hybrid ConvNeXt and Vision Transformer classifier with learned attention fusion recognises 33 item classes grouped into nine material families. "
    "An object–material cross-check never weakens a hazard call, Monte Carlo dropout and conformal family sets limit how often a hazard reaches recycling, "
    "and two low-cost sensors give an Organic Contamination Index. On 4,232 held-out images the first model reached 89.30% item accuracy, 94.14% "
    "family (routing) accuracy and 93.50% hazard recall; an improved training recipe reached 93.83%, 97.28% and 98.37% as reported by the training "
    "platform. Conformal sets cut hazard leakage from 6.5% to 1.57% at 7.8% of items sent for review. Sensor results use simulated data, and the "
    "physical prototype is the next stage."
)

ABBREVIATIONS = [
    ("AdamW", "Adam with Decoupled Weight Decay"), ("AI", "Artificial Intelligence"), ("AUC", "Area Under the Curve"),
    ("AUROC", "Area Under the Receiver Operating Characteristic Curve"), ("BOM", "Bill of Materials"),
    ("CAPS-FCL", "Contamination-Aware, Privacy-Preserving Smart Federated Classification and Learning"),
    ("CNN", "Convolutional Neural Network"), ("COCO", "Common Objects in Context"), ("CPU", "Central Processing Unit"),
    ("DCGAN", "Deep Convolutional Generative Adversarial Network"), ("ECE", "Expected Calibration Error"),
    ("FedAvg", "Federated Averaging"), ("FLOPs", "Floating-Point Operations"), ("GAN", "Generative Adversarial Network"),
    ("GELU", "Gaussian Error Linear Unit"), ("GPU", "Graphics Processing Unit"), ("Grad-CAM", "Gradient-weighted Class Activation Mapping"),
    ("HOG", "Histogram of Oriented Gradients"), ("HSV", "Hue, Saturation and Value"), ("IoT", "Internet of Things"),
    ("IoU", "Intersection over Union"), ("kNN", "k-Nearest Neighbours"), ("LDA", "Linear Discriminant Analysis"),
    ("LIME", "Local Interpretable Model-agnostic Explanations"), ("mAP", "mean Average Precision"), ("MC", "Monte Carlo"),
    ("MCC", "Matthews Correlation Coefficient"), ("MQ-135", "Metal-oxide air-quality gas sensor of model type MQ-135"),
    ("NLL", "Negative Log-Likelihood"), ("OCI", "Organic Contamination Index"), ("ONNX", "Open Neural Network Exchange"),
    ("OOD", "Out-of-Distribution"), ("PCA", "Principal Component Analysis"), ("RBF", "Radial Basis Function"),
    ("ROC", "Receiver Operating Characteristic"), ("SAM", "Segment Anything Model"),
    ("SCORE", "School of Computer Science Engineering and Information Systems"), ("SHAP", "SHapley Additive exPlanations"),
    ("SVM", "Support Vector Machine"), ("SWM", "Solid Waste Management"), ("TACO", "Trash Annotations in Context"),
    ("t-SNE", "t-Distributed Stochastic Neighbour Embedding"), ("TRL", "Technology Readiness Level"),
    ("UQ", "Uncertainty Quantification"), ("ViT", "Vision Transformer"), ("XAI", "Explainable Artificial Intelligence"),
    ("YOLO", "You Only Look Once"),
]

SYMBOLS = [
    ("x", "input image (or one cropped item)"), ("y", "true item class (one of C = 33 classes)"), ("C", "number of item classes (33)"),
    ("z<sub>i</sub>", "logit (raw score) of class i"), ("p<sub>i</sub>", "predicted probability of class i"),
    ("T", "temperature in temperature scaling; number of Monte Carlo dropout passes where stated"), ("ε", "label-smoothing amount (0.1)"),
    ("f<sub>s</sub>", "projected feature vector of backbone stream s (ConvNeXt or Vision Transformer)"),
    ("e<sub>s</sub>", "attention score of stream s"), ("α<sub>s</sub>", "attention weight of stream s (the weights add up to 1)"),
    ("M", "set of material classes that physically fit the recognised object"), ("H", "set of hazardous classes"),
    ("c", "confidence of the object-identification stage, between 0 and 1"),
    ("p̄", "average of the Monte Carlo dropout probability vectors"), ("U(x)", "normalised-entropy uncertainty of item x, between 0 and 1"),
    ("τ<sub>u</sub>", "uncertainty (review) threshold"), ("β", "share of calibration items sent for review (review budget)"),
    ("Q", "empirical quantile function"), ("m<sub>f</sub>(x)", "total probability that item x belongs to material family f"),
    ("s<sub>j</sub>", "nonconformity score of calibration item j"), ("n<sub>f</sub>", "number of calibration items whose true family is f"),
    ("α<sub>f</sub>, α<sub>H</sub>", "error level of family f, and the stricter error level of the hazardous family"),
    ("δ", "confidence parameter of the Beta-corrected guarantee"), ("k<sub>f</sub>", "rank of the calibration score used as threshold for family f"),
    ("q<sub>f</sub>", "conformal threshold of family f"), ("C(x)", "set of plausible material families for item x"),
    ("f<sub>m</sub>, f<sub>g</sub>", "scaled moisture score and scaled gas score, each between 0 and 1"),
    ("m<sub>dry</sub>, m<sub>wet</sub>", "moisture-sensor readings of a dry and a soaked reference"),
    ("R<sub>s</sub>, R<sub>0</sub>", "gas-sensor resistance now, and in clean air"), ("l<sub>min</sub>, l<sub>max</sub>", "limits of −ln(R<sub>s</sub>/R<sub>0</sub>) seen at commissioning"),
    ("σ(·)", "logistic (sigmoid) function"), ("β<sub>0</sub> … β<sub>3</sub>", "coefficients of the contamination-index model"),
    ("τ*", "contamination decision threshold"), ("θ<sub>k</sub>", "model parameters of sorting unit k"),
    ("n<sub>k</sub>, N", "images at unit k, and total number of images"), ("L<sub>H</sub>", "hazard leakage (share of hazards sent to a recycling bin)"),
    ("W", "review workload (share of items sent for manual review)"), ("F", "share of contamination decisions that flip when a sensor channel is lost"),
    ("η", "learning rate"), ("λ", "weight-decay coefficient"),
]
