# report_ch1.py - chapter 1 (introduction: objective, motivation, background concepts) of the project report
from report_equations import *  # noqa: F401,F403

TITLE = ("CHAPTER 1", "INTRODUCTION")

E_ATT = wrap(N("Attention") + D(R("Q") + R(", ") + R("K") + R(", ") + R("V")) + R(" = ") + N("softmax") +
             D(F(R("Q") + P(R("K"), N("T")), RAD(S(R("d"), R("k"))))) + R("V"))
E_SOFT = wrap(S(R("p"), R("i")) + R(" = ") + F(FN("exp", D(S(R("z"), R("i")))), SUM(R("j=1"), R("C"), FN("exp", D(S(R("z"), R("j")))))))
E_CE = wrap(R("L") + R(" = −") + SUM(R("i=1"), R("C"), S(R("q"), R("i")) + FN("log", S(R("p"), R("i")))) + R(",   ") +
            S(R("q"), R("i")) + R(" = (1 − ε)") + IND(R("i = y")) + R(" + ") + F(R("ε"), R("C")))
E_GELU = wrap(N("GELU") + D(R("x")) + R(" = x Φ(x) = ") + R("x ") + F(R("1"), R("2")) + D(R("1 + ") + N("erf") + D(F(R("x"), RAD(R("2")))), "[", "]"))
E_TEMP = wrap(S(R("p"), R("i")) + D(R("T")) + R(" = ") + F(FN("exp", D(F(S(R("z"), R("i")), R("T")))), SUM(R("j=1"), R("C"), FN("exp", D(F(S(R("z"), R("j")), R("T")))))))
E_SIG = wrap(R("σ(z) = ") + F(R("1"), R("1 + ") + P(R("e"), R("−z"))))


def blocks():
    B = []
    B += [
        ("p", "Waste management is one of the quiet infrastructure problems of the modern world. Every household, office and factory produces material that has to be "
              "collected, separated and either recovered or disposed of, and the quality of that separation decides how much of it can be recycled. The World Bank estimates "
              "that the world generated about 2.01 billion tonnes of municipal solid waste in 2016 and expects this to grow to about 3.40 billion tonnes a year by 2050, "
              "with at least a third of it not managed in an environmentally safe way (Kaza et al., 2018). Sorting is the step where a mixed stream becomes a set of clean "
              "streams, and it is still done largely by hand, which is slow, tiring and unsafe when the stream contains sharp or toxic items."),
        ("p", "Artificial Intelligence (AI) and computer vision now make it possible to place a camera above a conveyor belt or inside a bin and let a trained neural network "
              "decide where each item should go. Such systems have been built and deployed, but they share a set of weaknesses that this project sets out to address. "
              "They usually rely on one camera and one classifier, give one answer per item without saying how sure they are, work with a small number of broad categories, "
              "cannot sense contamination that the camera does not show, and treat a mistaken hazardous item (a battery or a medical sharp) like any other mistake. "
              "This report describes the design, implementation and evaluation of an edge-deployable waste-sorting pipeline, named CAPS-FCL (Contamination-Aware, "
              "Privacy-Preserving Smart Federated Classification and Learning), that tackles these weaknesses together."),
        ("h2", "1.1 Objective"),
        ("p", "The primary objective of this project is to develop an edge-deployable waste-sorting system that combines a hybrid deep-learning vision model with low-cost "
              "physical sensors, estimates its own uncertainty, protects against the dangerous mistake of sending hazardous waste to recycling, and can learn across many "
              "units without sharing raw images. The work is carried out in software on real held-out data first, with the physical prototype planned as the next stage."),
        ("p", "The specific objectives are:"),
        ("num", [
            "To build a detect-then-classify vision pipeline in which a YOLO object detector localises each item and a hybrid ConvNeXt and Vision Transformer classifier, "
            "joined by a learned attention layer, recognises 33 fine-grained item classes that are grouped into nine material families (routing bins).",
            "To report two kinds of accuracy separately, item accuracy for recognition and family accuracy for routing, together with hazard recall, so that a confusion "
            "between look-alike items that share a bin is not mistaken for a sorting error.",
            "To estimate how sure the system is of every prediction (Monte Carlo dropout and calibrated probabilities), to cross-check the object name against the material "
            "prediction without ever weakening a hazard call, and to give the operator a stated limit on how often a hazardous item may reach recycling.",
            "To design a two-sensor Organic Contamination Index (OCI), from a moisture sensor and a gas sensor, that detects residue the camera cannot see and keeps working when one "
            "sensor fails.",
            "To make decisions understandable with Grad-CAM, LIME and SHAP explanations and with a pixel-accurate item outline from the Segment Anything Model.",
            "To show with simulation that several sorting units can improve a shared model by federated averaging without sending images, and to extend the pipeline to video "
            "surveys of litter with one inventory entry per physical item.",
        ]),
        ("h2", "1.2 Motivation"),
        ("p", "Four observations motivated the project. The first is that recycling quality depends on purity. A single contaminated item, such as a food-soiled pizza box, can "
              "lower the value of a whole bale of cardboard. The camera sees the cardboard but not the grease, so a vision-only sorter will pass the item. Cheap sensors "
              "for moisture and for gas given off by decaying organic matter can add the missing information, and an index built from them can be calibrated against known "
              "contamination levels."),
        ("p", "The second observation is that some mistakes are far more costly than others. Putting a plastic bottle in the paper bin is a small loss. Putting a lithium battery "
              "in a recycling stream risks fires in the sorting plant, and electronic waste is a growing hazard in its own right: the Global E-waste Monitor reports 62 million tonnes "
              "of electronic waste in 2022 and notes that it is rising five times faster than documented recycling (United Nations Institute for Training and Research, 2024). A useful sorter therefore has to know "
              "when it is unsure, send unsure hazardous items to a person, and give a measurable limit on how often a hazard slips through."),
        ("p", "The third observation is that many items look alike. Fig. {F:look} shows pairs of classes that even a person would find hard to separate from a photograph: "
              "cardboard boxes and cardboard packaging, and steel and aluminium food cans. A system with a handful of broad categories hides this difficulty, whereas a system "
              "with dozens of classes meets it directly and has to separate recognition errors that do not matter (the two items share a bin) from routing errors that do."),
        ("fig", "look", "lookalike", "Look-Alike Classes From the Test Set: Cardboard Boxes and Packaging (Top), Steel and Aluminium Food Cans (Bottom)", 12.0),
        ("p", "The fourth observation concerns deployment and privacy. A sorting unit at the source of waste must decide within a fraction of a second, often without a reliable "
              "network connection, and the images it sees can show people, homes and documents. Running the models on the edge device and sharing only model parameters, never "
              "images, between units addresses both latency and privacy. Federated learning provides the mechanism for this."),
        ("h2", "1.3 Background"),
        ("p", "This section introduces the concepts on which the rest of the report depends. Each concept is described briefly and linked to the place in the project where it is used."),
        ("h3", "1.3.1 Waste Streams, Contamination and Hazardous Items"),
        ("p", "A recycling stream is made of material families such as plastic, paper, cardboard, glass and metal. A sorting facility acts on the family, because each family has its "
              "own bin and its own processing. Contamination is any foreign matter, mostly food or liquid residue, that makes an item unsuitable for recycling even though its "
              "material is recyclable. Hazardous items are a separate family: batteries, electronic waste and medical waste need special handling by law and by safety. In India "
              "these are governed by rules such as the Solid Waste Management Rules, 2016, the Bio-Medical Waste Management Rules, 2016, the E-Waste (Management) Rules, 2022 and the "
              "Battery Waste Management Rules, 2022 (Ministry of Environment, Forest and Climate Change, 2016a, 2016b, 2022a, 2022b)."),
        ("h3", "1.3.2 Convolutional Networks and ConvNeXt"),
        ("p", "A convolutional neural network (CNN) learns small filters that slide over an image and respond to edges, textures and shapes, building up from local patterns to whole "
              "objects. Its two built-in assumptions, that nearby pixels are related and that a pattern means the same thing wherever it appears, make it efficient and well suited to "
              "surface texture, which carries much of the signal in material recognition (foil crinkles, cardboard fibres). ConvNeXt modernises a standard residual network with "
              "larger kernels, layer normalisation and GELU activations, and matches transformer accuracy while staying a pure convolutional design (Liu et al., 2022). "
              "The project uses ConvNeXt-Tiny with 27.8 million parameters."),
        ("h3", "1.3.3 Transformers and Attention"),
        ("p", "A Vision Transformer (ViT) cuts an image into patches and lets every patch look at every other patch through self-attention (Dosovitskiy et al., 2021; Vaswani et al., 2017). "
              "Attention computes, for each position, a weighted average of the values of all positions, with weights given by the similarity of queries and keys, as in "
              "{E:att}. In {E:att} the variables Q, K and V are the query, key and value matrices, d<sub>k</sub> is the dimension of the keys, and the division by the square root of d<sub>k</sub> keeps "
              "the dot products in a range where the softmax is well behaved. Because attention is global from the first layer, a transformer captures shape and context that "
              "a small convolution misses. The project uses ViT-Small with 16 × 16 patches and 21.7 million parameters."),
        ("eq", "att", E_ATT),
        ("h3", "1.3.4 Transfer Learning and Fine-Tuning"),
        ("p", "Training a large network from scratch needs far more images than the 28,000 available here; training the hybrid from scratch was in fact tried once in this project "
              "on an earlier dataset and reached only 49% accuracy. Transfer learning starts from weights learned on a large dataset (ImageNet) and adapts them. Fine-tuning adapts all layers, "
              "which is powerful but can distort good pretrained features if the new classifier layers are still random and send large noisy gradients backwards. "
              "Training the new layers first and then fine-tuning with a lower learning rate for the backbone protects the pretrained features (Kumar et al., 2022). "
              "This idea became the improved training recipe of Chapter 4."),
        ("h3", "1.3.5 Training Objectives and Activation Functions"),
        ("p", "A classifier outputs one raw score, called a logit z<sub>i</sub>, for each of the C classes. The softmax function in {E:soft} turns the logits into probabilities p<sub>i</sub> "
              "that are positive and add up to one. The training loss is the cross-entropy between the predicted probabilities and a target distribution. With label smoothing "
              "(Szegedy et al., 2016), the target puts the share (1 − ε) on the true class y and spreads ε equally over all classes, as in {E:ce}, where the indicator "
              "1[i = y] is one for the true class and zero otherwise and ε = 0.1 in this project. Smoothing stops the network from becoming over-confident between near-duplicate classes."),
        ("eq", "soft", E_SOFT),
        ("eq", "ce", E_CE),
        ("p", "Hidden layers need a non-linear activation. The Gaussian Error Linear Unit in {E:gelu} multiplies its input x by the probability Φ(x) that a standard normal variable is "
              "smaller than x; erf is the error function. Unlike the rectified linear unit it is smooth and does not leave neurons permanently switched off, and it is the activation "
              "used by both pretrained backbones (Hendrycks &amp; Gimpel, 2016). The optimiser is AdamW, an adaptive method whose weight decay is applied directly to the weights "
              "and not through the gradient (Loshchilov &amp; Hutter, 2019)."),
        ("eq", "gelu", E_GELU),
        ("h3", "1.3.6 Class Imbalance"),
        ("p", "When some classes have many more images than others, a classifier drifts towards the common classes. Common remedies are to draw rare classes more often during "
              "training (a balanced sampler), to give rare classes a larger weight in the loss, or to shift the logits after training by the logarithm of the class frequencies "
              "(logit adjustment; Menon et al., 2021). In this project the largest class has 11.6 times the images of the smallest, and Chapter 7 reports what correcting the imbalance did."),
        ("h3", "1.3.7 Calibration and Uncertainty"),
        ("p", "A model is calibrated when, among all predictions made with 80% confidence, about 80% are right. Modern networks are often over-confident, and training choices can also make "
              "them under-confident (Guo et al., 2017). Temperature scaling divides all logits by one number T fitted on validation data, as in {E:temp}; it changes the confidence "
              "but never the predicted class. Fig. {F:temp} shows how a value of T below one sharpens the probabilities and a value above one flattens them."),
        ("eq", "temp", E_TEMP),
        ("fig", "temp", "temp", "Effect of the Temperature T on the Probabilities Obtained From the Same Five Logits", 9.5),
        ("p", "Uncertainty can also be estimated by running the model many times with dropout switched on and looking at how much the answers disagree; this Monte Carlo "
              "dropout approximates Bayesian inference (Gal &amp; Ghahramani, 2016). A simpler baseline is the largest predicted probability (Hendrycks &amp; Gimpel, 2017). "
              "A selective classifier may decline to answer when it is unsure, which in a sorter means sending the item to a person (Geifman &amp; El-Yaniv, 2017)."),
        ("h3", "1.3.8 Conformal Prediction"),
        ("p", "Conformal prediction turns any classifier into one that outputs a set of plausible answers with a guaranteed chance of containing the truth (Vovk et al., 2005; "
              "Angelopoulos &amp; Bates, 2023). A threshold is calibrated on held-out labelled data so that, for a chosen error level α, the true answer is inside the set at "
              "least 1 − α of the time for new items that resemble the calibration data. Calibrating separately for each class (Vovk, 2012; Sadinle et al., 2019) lets the "
              "operator choose a stricter level for the class that matters most. This project calibrates one threshold per material family and uses a much stricter level for the "
              "hazardous family."),
        ("h3", "1.3.9 Object Detection, Tracking and Segmentation"),
        ("p", "An object detector such as You Only Look Once (YOLO) finds objects in one pass and reports a box, a name and a confidence for each (Redmon et al., 2016). "
              "Accuracy is measured by mean average precision (mAP) at a chosen overlap (intersection over union) between predicted and true boxes, as defined by the Common Objects in Context benchmark (Lin et al., 2014). In video, a tracker such as ByteTrack "
              "gives each object a persistent number so that it can be counted once (Zhang et al., 2022). The Segment Anything Model draws a pixel-accurate outline "
              "inside a box (Kirillov et al., 2023), and its mobile version suits edge devices (Zhang et al., 2023)."),
        ("h3", "1.3.10 Explainable Artificial Intelligence"),
        ("p", "Explanations build trust and help find faults. Grad-CAM colours the image regions that raised the score of a class (Selvaraju et al., 2017). Local Interpretable "
              "Model-agnostic Explanations (LIME) hide parts of the image and watch how the prediction changes (Ribeiro et al., 2016). SHapley Additive exPlanations (SHAP) share a "
              "prediction among the inputs so that the shares add up exactly (Lundberg &amp; Lee, 2017); here SHAP explains the contamination score sensor by sensor."),
        ("h3", "1.3.11 Sensor Fusion and Logistic Regression"),
        ("p", "Sensor fusion combines readings from several sensors so that the result is better than any one alone (Hussain et al., 2024). The contamination index uses logistic "
              "regression, a linear model passed through the logistic function σ in {E:sig}, which maps any number to the range between 0 and 1 and so can be read as a probability of "
              "contamination. Its coefficients are learned from labelled calibration data, and a decision threshold is chosen from the receiver operating characteristic (ROC) curve."),
        ("eq", "sig", E_SIG),
        ("h3", "1.3.12 Federated Learning and Edge Deployment"),
        ("p", "In federated learning, each unit trains on its own data and sends only model parameters to a server, which averages them into a better shared model "
              "(McMahan et al., 2017). Edge deployment means that the trained models run on the small computer inside the unit. Exporting a model to the Open "
              "Neural Network Exchange (ONNX) format lets it run on different runtimes and devices without the original training framework."),
        ("h3", "1.3.13 Evaluation Metrics"),
        ("p", "Accuracy is the share of correct predictions. For each class, precision is the share of predictions of that class that are right and recall is the share of the "
              "true members that are found; the F1 score is their harmonic mean and the macro F1 averages it over classes so that rare classes count equally. Cohen's kappa and "
              "the Matthews correlation coefficient correct accuracy for chance. Top-k accuracy counts a prediction as right if the true class is among the k most likely. "
              "The ROC and precision–recall curves summarise a classifier over all thresholds, and their areas (AUC) are single numbers. Calibration is measured by the expected "
              "calibration error (ECE). Chapter 7 adds two operational metrics defined for sorting: hazard leakage and review workload."),
        ("h2", "1.4 Organisation of the Report"),
        ("p", "Chapter 2 states the problem, the goals and scope, reviews the literature and patents, and identifies the research gap. Chapter 3 gives the technical specification: "
              "requirements, datasets, hardware and software. Chapter 4 describes the design approach, including the equations of every mechanism, the standards followed and the "
              "constraints and alternatives. Chapter 5 gives the schedule and milestones. Chapter 6 demonstrates the working system on real images, video and simulated sensors. "
              "Chapter 7 reports and discusses the results and the cost analysis. Chapter 8 summarises the work and recommends future steps, and Chapter 9 lists the references. "
              "Appendix A holds sample code and screen captures."),
    ]
    return B
