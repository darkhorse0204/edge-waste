# report_ch4.py - chapter 4 (design approach and details: architecture, methods with equations, standards, constraints)
from report_equations import *  # noqa: F401,F403

TITLE = ("CHAPTER 4", "DESIGN APPROACH AND DETAILS")

s = S
E_ES = wrap(s(R("e"), R("s")) + R(" = ") + s(R("W"), R("2")) + R(" · ") + N("GELU") + D(s(R("W"), R("1")) + s(R("f"), R("s")) + R(" + ") + s(R("b"), R("1"))) + R(" + ") + s(R("b"), R("2")))
E_AL = wrap(s(R("α"), R("s")) + R(" = ") + F(FN("exp", D(s(R("e"), R("s")))), SUM(R("s′"), "", FN("exp", D(s(R("e"), R("s′")))))))
E_FUSE = wrap(R("h = ") + D(s(R("α"), N("cnx")) + s(R("f"), N("cnx")) + R(" ;  ") + s(R("α"), N("vit")) + s(R("f"), N("vit")), "[", "]"))
E_ADAMW = wrap(EQARR(
    s(R("m"), R("t")) + R(" = ") + s(R("β"), R("1")) + s(R("m"), R("t−1")) + R(" + (1 − ") + s(R("β"), R("1")) + R(")") + s(R("g"), R("t")),
    s(R("v"), R("t")) + R(" = ") + s(R("β"), R("2")) + s(R("v"), R("t−1")) + R(" + (1 − ") + s(R("β"), R("2")) + R(")") + P(s(R("g"), R("t")), R("2")),
    s(R("θ"), R("t")) + R(" = ") + s(R("θ"), R("t−1")) + R(" − η") + D(F(HAT(s(R("m"), R("t"))), RAD(HAT(s(R("v"), R("t")))) + R(" + ε")) + R(" + λ") + s(R("θ"), R("t−1")))))
E_LOGADJ = wrap(SP(R("z"), R("i"), R("′")) + R(" = ") + s(R("z"), R("i")) + R(" + τ ") + FN("log", s(R("n"), R("i"))))
E_PRIOR = wrap(SP(R("p"), R("i"), R("′")) + R(" = ") + F(s(R("w"), R("i")) + s(R("p"), R("i")), SUM(R("j"), "", s(R("w"), R("j")) + s(R("p"), R("j")))) + R(",   ") +
               s(R("w"), R("i")) + R(" = ") + EQARR(R("1   ") + T("if ") + R("i ∈ M ") + T("or ") + R("i ∈ H"), R("0.15   ") + T("otherwise")))
E_MC = wrap(BAR(R("p")) + R(" = ") + F(R("1"), R("T")) + SUM(R("t=1"), R("T"), N("softmax") + D(s(R("f"), R("θ")) + D(R("x") + R("; ") + s(N("dropout"), R("t"))))))
E_U = wrap(R("U(x) = ") + F(R("−") + SUM(R("c"), "", s(BAR(R("p")), R("c")) + FN("log", s(BAR(R("p")), R("c")))), FN("log", R("C"))))
E_TAU = wrap(s(R("τ"), R("u")) + R(" = ") + s(R("Q"), R("1−β")) + D(R("{ U(") + s(R("x"), R("j")) + R(") : j ∈ calibration set }")))
E_FAM = wrap(s(R("m"), R("f")) + R("(x) = ") + SUM(R("i ∈ f"), "", s(R("p"), R("i")) + R("(x)")))
E_RANK = wrap(s(R("q"), R("f")) + R(" = ") + s(R("s"), D(s(R("k"), R("f")))) + R(",   ") + s(R("k"), R("f")) + R(" = ") +
              D(D(s(R("n"), R("f")) + R(" + 1")) + D(R("1 − ") + s(R("α"), R("f"))), "⌈", "⌉"))
E_BRANK = wrap(s(R("k"), R("f")) + R(" = ") + LIMLOW(N("min"), R("k")) + R(" { k : ") + SP(R("B"), R("k, ") + s(R("n"), R("f")) + R(" + 1 − k"), R("−1")) + D(R("δ")) + R(" ≥ 1 − ") + s(R("α"), R("f")) + R(" }"))
E_SET = wrap(R("C(x) = { f : 1 − ") + s(R("m"), R("f")) + R("(x) ≤ ") + s(R("q"), R("f")) + R(" }"))
E_BOUND = wrap(EQARR(
    N("Pr") + D(T("a true hazard reaches a non-hazardous gate"), "[", "]") + R(" ≤ ") + s(R("α"), R("H")),
    N("Pr") + D(T("an item of family f goes automatically to a wrong gate"), "[", "]") + R(" ≤ ") + s(R("α"), R("f"))))
E_FM = wrap(s(R("f"), R("m")) + R(" = ") + N("clip") + D(F(s(R("m"), N("raw")) + R(" − ") + s(R("m"), N("dry")), s(R("m"), N("wet")) + R(" − ") + s(R("m"), N("dry"))) + R(", 0, 1")))
E_FG = wrap(s(R("f"), R("g")) + R(" = ") + N("clip") + D(F(R("−ln") + D(F(s(R("R"), R("s")), s(R("R"), R("0")))) + R(" − ") + s(R("l"), N("min")), s(R("l"), N("max")) + R(" − ") + s(R("l"), N("min"))) + R(", 0, 1")))
E_OCI = wrap(N("OCI") + R(" = σ") + D(s(R("β"), R("0")) + R(" + ") + s(R("β"), R("1")) + s(R("f"), R("m")) + R(" + ") + s(R("β"), R("2")) + s(R("f"), R("g")) + R(" + ") + s(R("β"), R("3")) + s(R("f"), R("m")) + s(R("f"), R("g"))))
E_OCI1 = wrap(N("OCI") + R(" = σ") + D(s(R("β"), R("0,m")) + R(" + ") + s(R("β"), R("1,m")) + s(R("f"), R("m"))) + R("   or   σ") + D(s(R("β"), R("0,g")) + R(" + ") + s(R("β"), R("1,g")) + s(R("f"), R("g"))))
E_THR = wrap(P(R("τ"), R("*")) + R(" = ") + LIMLOW(N("argmin"), R("τ : TPR(τ) ≥ 0.95")) + N(" FPR") + D(R("τ")))
E_FED = wrap(s(R("θ"), N("global")) + R(" = ") + SUM(R("k=1"), R("K"), F(s(R("n"), R("k")), R("N")) + s(R("θ"), R("k"))))

PARAMS = [("ConvNeXt-Tiny backbone", "27.82 M"), ("Vision Transformer Small backbone", "21.67 M"), ("Two projections (linear, layer norm, GELU)", "0.59 M"),
          ("Attention fusion", "0.13 M"), ("Classifier head", "0.54 M"), ("Total", "50.75 M")]


def blocks():
    B = []
    B += [
        ("h2", "4.1 Design Approach and Methods"),
        ("p", "The design follows one principle: put safety and honesty around the classifier. The classifier is one component, and the mechanisms around it decide what to do "
              "with its answer, how sure to be, and what to do when it is unsure. The design was built stage by stage, with every stage measured on held-out data before the next was added."),
        ("h3", "4.1.1 Overall Architecture"),
        ("p", "{F:sys} shows the architecture. A camera frame goes to an object detector, which gives a box, an object name and a confidence, and to the material classifier, which gives "
              "33 class probabilities. The object-identity cross-check narrows the material guess when that is safe. The uncertainty estimator and the family-set router measure "
              "how sure the system is, and the moisture and gas sensors give the contamination score. The decision engine combines the three signals and drives the actuator. "
              "{F:seq} shows the same flow for one item as a sequence of messages."),
        ("fig", "sys", "sys", "Overall Architecture: Cross-Check, Uncertainty Gating, Calibrated Family-Set Router, Contamination Index and Decision Engine", 14.0, 17.0),
        ("fig", "seq", "seq", "Sequence of Processing Steps for One Item", 14.4),
        ("h3", "4.1.2 Object Detection and the Object-Identity Prior"),
        ("p", "The detector is YOLO26n (2.4 million parameters), fine-tuned on TACO at 640 pixels. Two versions were trained: a single-class “waste item” locator and an 18-class "
              "object identifier whose classes include bottle, can, cup, carton, cigarette and plastic bag. The object name carries information about the material: a can is "
              "not made of cardboard, and a straw is not glass. A lookup table lists, for each object name, the materials that physically fit it. The classifier's probabilities are "
              "multiplied by a weight w<sub>i</sub> that is 1 for fitting materials and 0.15 for materials that do not fit, and are then renormalised ({E:prior}). In {E:prior} p<sub>i</sub> is the "
              "classifier's probability for class i, M is the set of materials that fit the recognised object, and H is the set of hazardous classes."),
        ("eq", "prior", E_PRIOR),
        ("p", "Two design decisions matter here. First, implausible materials are down-weighted, not forbidden, so that a surprising but correct reading can still win if the classifier "
              "is confident; a hard mask would make the system unable to report a mislabelled object, which is worse than the error it prevents. Second, hazardous classes always have weight 1, "
              "so the object stage can never talk the classifier out of a hazard call. If the two stages still disagree after re-weighting, the contradiction is reported as its own signal and "
              "the item goes to manual review, and a hazard call is never treated as a contradiction. For the COCO-pretrained recogniser, which names everyday objects such as bottle, cup or "
              "banana, the mass given to a “certain” mapping is 0.95 scaled by the identity confidence, so a low-confidence name has little effect."),
        ("h3", "4.1.3 Hybrid Classifier"),
        ("p", "The material classifier ({F:hybrid}) uses two pretrained backbones: ConvNeXt-Tiny, a convolutional stream that is strong on local texture, and Vision Transformer "
              "Small with 16 × 16 patches, an attention stream that is strong on global shape and context. Each backbone gives a pooled feature vector (768 and 384 values), and each is "
              "projected to 512 values by a linear layer, layer normalisation and GELU. Layer normalisation is used instead of batch normalisation because it does not depend on batch "
              "statistics and so behaves the same at the training batch size of 32 and at an edge batch size of 1."),
        ("p", "An attention module then decides how much to trust each stream for the current image. In {E:es}, f<sub>s</sub> is the projected feature vector of stream s (ConvNeXt or Vision "
              "Transformer), W<sub>1</sub>, b<sub>1</sub>, W<sub>2</sub> and b<sub>2</sub> are learned weights and offsets, and e<sub>s</sub> is the attention score of the stream. The softmax in {E:al} turns the two "
              "scores into weights α<sub>s</sub> that are positive and add up to one. The fused vector in {E:fuse} places the two weighted streams side by side (1,024 values), and a head of "
              "dropout, a linear layer from 1,024 to 512 values, GELU, dropout and a final linear layer to 33 values gives the logits. The family probability is the sum of the "
              "probabilities of the items in the family."),
        ("eq", "es", E_ES), ("eq", "al", E_AL), ("eq", "fuse", E_FUSE),
        ("fig", "hybrid", "hybrid", "Architecture of the Hybrid ConvNeXt and Vision Transformer Classifier With Attention Fusion", 8.8, 19.0),
        ("tbl", "params", "Parameters of the Hybrid Classifier", ["Component", "Parameters"], PARAMS, [9.0, 4.0], {"align_cols": ["l", "c"]}),
        ("h3", "4.1.4 Training Recipe"),
        ("p", "The classifier is trained with cross-entropy loss with label smoothing of 0.1 ({E:ce} of Chapter 1) and the AdamW optimiser. {E:adamw} shows one AdamW step: g<sub>t</sub> is the "
              "gradient, m<sub>t</sub> and v<sub>t</sub> are running averages of the gradient and of its square with decay rates β<sub>1</sub> and β<sub>2</sub>, the hats mark bias-corrected values, "
              "η is the learning rate, ε is a small constant for numerical stability and λ is the weight-decay coefficient, which is applied to the weights directly. "
              "A one-cycle schedule warms the learning rate up for the first epoch and then lowers it with a cosine curve. {T:hyper} lists the settings of the first model."),
        ("eq", "adamw", E_ADAMW),
        ("tbl", "hyper", "Training Settings of the First Model", ["Setting", "Value", "Reason"], [
            ("Optimiser", "AdamW, peak learning rate 3e-4, weight decay 0.05", "adaptive steps suit two pretrained backbones and a new head"),
            ("Schedule", "one-cycle: one warm-up epoch, then cosine decay", "warm-up protects pretrained weights"),
            ("Loss", "cross-entropy, label smoothing 0.1", "limits over-confidence between look-alike classes"),
            ("Imbalance handling", "balanced sampler and inverse-frequency loss weights", "later found to correct twice (Section 7.5)"),
            ("Batch size and epochs", "32 and 15, early-stopping patience 5", "best epoch was the last one"),
            ("Precision and clipping", "16-bit mixed precision, gradient norm clipped at 1.0", "speed and stability"),
            ("Seed", "42", "reproducibility"),
            ("Hardware", "Google Colab T4, about 4 minutes per epoch", "the laptop needs about 55 minutes per epoch"),
        ], [3.0, 5.7, 5.8]),
        ("p", "The analysis of this model (Chapter 7) found that all 50 million parameters had been fine-tuned at the same peak learning rate from the first step, while the "
              "classifier head was still random. This distorts pretrained features (Kumar et al., 2022). The improved recipe in {T:recipe} freezes the backbones for the first two epochs "
              "so that the head learns first, then trains the backbones ten times more slowly than the new layers, and corrects class imbalance once. Where a model has already been "
              "trained with a double correction, the logits can be adjusted after training ({E:logadj}), in which n<sub>i</sub> is the number of training images of class i and τ is a strength "
              "chosen on validation data; this adds back part of the class frequency that the double correction removed (Menon et al., 2021)."),
        ("eq", "logadj", E_LOGADJ),
        ("tbl", "recipe", "Changes in the Improved Training Recipe", ["Change", "First model", "Improved recipe"], [
            ("Order of training", "all layers from the first step", "backbones frozen for the first 2 epochs (linear probe), then fine-tuned"),
            ("Learning rate", "3e-4 for all layers", "3e-5 for backbones, 3e-4 for new layers"),
            ("Imbalance correction", "sampler and loss weights", "sampler only"),
            ("Epochs", "15", "25 with early stopping (stopped after 12 on the training platform)"),
        ], [3.5, 4.5, 6.5]),
        ("h3", "4.1.5 Uncertainty and the Two-Tier Hazard Gate"),
        ("p", "Uncertainty is estimated by Monte Carlo dropout. The classifier is run T = 25 times with its dropout layers active (the normalisation layers stay fixed), and the "
              "probability vectors are averaged in {E:mc}. The uncertainty U(x) in {E:u} is the entropy of the averaged vector divided by log C, the largest possible entropy for C = 33 "
              "classes, so it runs from 0 (completely sure) to 1 (all classes equally likely). The two backbones contain no active dropout, so repeating only the head gives exactly "
              "the same result as repeating the whole network at about one hundredth of the cost; this was checked to give identical class predictions."),
        ("eq", "mc", E_MC), ("eq", "u", E_U),
        ("p", "The gate has two tiers ({F:gate}). A predicted hazard is acted on immediately, by sending it to the hazardous bin, only when U(x) is below the threshold τ<sub>u</sub>. At or "
              "above the threshold, the same prediction goes to priority manual review. In both cases the item stays out of ordinary recycling and compost. The threshold is not fixed. "
              "It is set from calibration data so that a chosen share β of calibration items goes to review ({E:tau}), where Q is the quantile function. With β = 0.10 about one item in ten is "
              "sent to a person, and the operator changes β to match the people available."),
        ("eq", "tau", E_TAU),
        ("fig", "gate", "gate", "Two-Tier Uncertainty-Gated Hazard Routing", 13.0),
        ("h3", "4.1.6 Conformal Family-Set Routing"),
        ("p", "The gate above acts when the single most likely class is hazardous. A hazardous item whose most likely class is not hazardous, for example a battery read as office "
              "paper, would not reach it, and no threshold would limit how often that happens. Conformal family-set routing addresses this. First, the item probabilities are added "
              "up into family totals ({E:fam}), so that evidence spread over several hazardous classes (battery, electronic waste, medical waste) stays together. In {E:fam}, m<sub>f</sub>(x) is the total probability that "
              "item x belongs to family f."),
        ("eq", "fam", E_FAM),
        ("p", "Second, a threshold is calibrated for each family on held-out labelled items. For family f the nonconformity scores s<sub>j</sub> = 1 − m<sub>f</sub>(x<sub>j</sub>) of the n<sub>f</sub> calibration "
              "items whose true family is f are sorted, and the threshold q<sub>f</sub> is the score of rank k<sub>f</sub> ({E:rank}), where α<sub>f</sub> is the share of family-f items the operator accepts to "
              "miss and the brackets round up. The hazardous family has its own, stricter level α<sub>H</sub>. The standard rank gives a guarantee on average over calibration sets. The Beta-corrected "
              "rank in {E:brank}, in which B<sup>−1</sup> is the inverse cumulative distribution function of the Beta distribution, gives a guarantee for the specific calibration set in use with "
              "probability at least 1 − δ, which suits a deployed machine."),
        ("eq", "rank", E_RANK), ("eq", "brank", E_BRANK),
        ("p", "Third, the set of plausible families is formed ({E:set}) and the route is chosen from the set, not from the single highest probability: (a) if hazardous is in the set and is "
              "the only family and U(x) < τ<sub>u</sub>, the item goes to the hazardous bin; (b) if hazardous is in the set in any other case, it goes to priority review and stays out of every "
              "recycling bin; (c) if hazardous is not in the set, the set has one family and U(x) < τ<sub>u</sub>, it goes automatically to that family's bin; (d) otherwise it goes to manual review "
              "({F:conf}). A true hazard can reach a recycling bin only if the hazardous family is missing from its set, and an item of family f can be sent automatically to a wrong bin only if f is "
              "missing from its set. Therefore, for items that resemble the calibration items, {E:bound} holds with probability at least 1 − δ over the calibration set."),
        ("eq", "set", E_SET), ("eq", "bound", E_BOUND),
        ("fig", "conf", "conf_d", "Calibrated Routing by Sets of Material Families", 13.0),
        ("p", "The guarantee is about items that look like the calibration items. Under damaged images the uncertainty gate adds protection (Section 7.9). To recalibrate after a model update or a "
              "change of site, only a new labelled calibration sample is needed; no retraining is required."),
        ("h3", "4.1.7 Organic Contamination Index"),
        ("p", "A greasy pizza box is cardboard but cannot be recycled, and the camera cannot see grease. Two sensors can: a capacitive moisture sensor and a metal-oxide gas "
              "sensor of the MQ-135 type that responds to volatile compounds from decaying organic matter. Each reading is scaled to the range 0 to 1 with fixed calibration "
              "anchors that are set once when the sensors are commissioned and are not recalculated from live readings. In {E:fm}, m<sub>raw</sub> is the live moisture reading and m<sub>dry</sub> and "
              "m<sub>wet</sub> are the readings of a dry and a soaked reference. In {E:fg}, R<sub>s</sub> is the present resistance of the gas sensor, R<sub>0</sub> its resistance in clean air, and "
              "l<sub>min</sub> and l<sub>max</sub> the smallest and largest values of −ln(R<sub>s</sub>/R<sub>0</sub>) seen at commissioning."),
        ("eq", "fm", E_FM), ("eq", "fg", E_FG),
        ("p", "Three logistic-regression models are fitted from calibration data ({F:oci}). The combined model in {E:oci} uses both scores and an interaction term, with coefficients "
              "β<sub>0</sub> to β<sub>3</sub> and the logistic function σ of {E:sig}, so that an item that is both wet and smelly counts for more than the two signals added separately. "
              "The two fallback models in {E:oci1} each use one sensor. At run time the model that matches the available sensors is chosen directly, instead of putting a default value for "
              "a missing sensor into the combined model, and if both sensors are lost the item goes to manual review. The decision threshold in {E:thr} is the one with the lowest false "
              "positive rate among those that reach a true positive rate (sensitivity) of 95%, because missing a contamination event costs more than an unnecessary check. It is chosen "
              "separately for each of the three models."),
        ("eq", "oci", E_OCI), ("eq", "oci1", E_OCI1), ("eq", "thr", E_THR),
        ("fig", "oci", "oci_d", "Two-Channel Organic Contamination Index With Model Choice by Sensor Availability", 14.4),
        ("h3", "4.1.8 Decision Engine"),
        ("p", "The decision engine combines the fused classification, the uncertainty, the family set and the contamination score into one decision for each item ({F:dec} and {T:rules}). "
              "In the working system the actuator is a mock that logs which gate and direction it would use; in a physical unit it would drive a conveyor and a servo diverter. "
              "Hazard beats ambiguity, but not blindly: an uncertain hazard call is treated as noise that landed on the most alarming label by chance, as was observed on "
              "out-of-distribution crops, and goes to priority review instead of triggering the hazardous gate."),
        ("tbl", "rules", "Decision Rules of the Engine", ["Condition", "Action"], [
            ("Hazardous family is the only family in the set and U(x) is below τ<sub>u</sub>", "hazardous bin"),
            ("Hazardous family is in the set in any other case", "priority manual review, never a recycling bin"),
            ("Object and material contradict each other after re-weighting", "manual review"),
            ("Set is empty or has several families, or U(x) is at or above τ<sub>u</sub>", "manual review"),
            ("Contamination index above its threshold on a recyclable item", "contamination reject"),
            ("Otherwise (one family, sure, not contaminated)", "bin of that family"),
        ], [9.5, 5.0]),
        ("fig", "dec", "dec", "Flow of the Decision Engine", 11.5, 16.0),
        ("h3", "4.1.9 Explainability and Segmentation"),
        ("p", "Grad-CAM is attached to the last stage of the ConvNeXt stream; the transformer stream has no equivalent spatial feature map, so the heat map explains the ConvNeXt view, "
              "which carries most of the attention weight. LIME perturbs superpixels of the image and treats the whole classifier as a black box, and agreement between the two methods is "
              "evidence that the decision rests on the item. SHAP explains the contamination score per sensor; for a linear model it is exact, and the contributions add up to the score "
              "with an error below 10<sup>−6</sup>. The box-prompted Segment Anything Model (the mobile version, about 40 MB) turns the detector's rectangle into a pixel-accurate outline so that "
              "background pixels do not reach the classifier."),
        ("h3", "4.1.10 Federated Learning"),
        ("p", "Several sorting units each train a local copy of the shared classifier on their own images and send only the updated parameters to an aggregation server ({F:fed}). "
              "The server computes the weighted average in {E:fed}, where θ<sub>k</sub> are the parameters of unit k, n<sub>k</sub> is its number of images, N is the total number of images over all K units, "
              "and θ<sub>global</sub> is the new shared model that is sent back to every unit. Units with more experience have more influence, and no image leaves any unit. The implementation "
              "(federated averaging) supports equal and unequal data splits across simulated units."),
        ("eq", "fed", E_FED),
        ("fig", "fed", "fed_d", "Federated Averaging Across Sorting Units: Only Model Parameters Are Shared", 10.5, 14.5),
        ("h3", "4.1.11 Video Survey, Synthetic Images and Edge Export"),
        ("p", "For video of a scene with many items, each detection receives a persistent track number from ByteTrack (Zhang et al., 2022), the object name of a track is chosen by "
              "a confidence-weighted vote over its frames, and the material classification is done once on the most informative frame (for example the frame with the largest box). "
              "The result is one inventory entry per physical item, however many frames it appears in. For classes with few images, a deep convolutional generative adversarial "
              "network (Radford et al., 2016) can create extra training images; this was implemented and tried on the textile class, but the generated images are small and soft and are "
              "not used in the reported models. The classifier and the detector are exported to ONNX and checked against the original models: the largest difference in the "
              "classifier outputs is 9.4 × 10<sup>−6</sup>, and the files are 203 MB and 9.3 MB."),
        ("h2", "4.2 Codes and Standards"),
        ("p", "The project follows the standards and rules listed in {T:std}. The Indian rules are those that define the hazardous family and the sorting duties the system "
              "is meant to support; they are not claimed as certifications."),
        ("tbl", "std", "Codes, Standards and Rules Followed", ["Standard or rule", "How it is followed in the project"], [
            ("ISO/IEC/IEEE 29148 (requirements engineering)", "requirements written with identifiers, verification method and priority (Chapter 3)"),
            ("Solid Waste Management Rules, 2016 (India)", "source separation into dry, wet and hazardous streams motivates the family-level routing"),
            ("Bio-Medical Waste Management Rules, 2016; E-Waste (Management) Rules, 2022; Battery Waste Management Rules, 2022", "define the medical, electronic and battery wastes that make up the hazardous family and must never go to recycling"),
            ("Open Neural Network Exchange (ONNX) format", "model export with a numerical check against the original model"),
            ("Python PEP 8 style and one-line header comments", "readable, uniform source files"),
            ("NASA Technology Readiness Level scale", "the status is stated as level 3, an experimental proof of concept"),
            ("APA 7th edition", "citation and reference style of this report"),
        ], [6.0, 8.5]),
        ("h2", "4.3 Constraints, Alternatives and Tradeoffs"),
        ("p", "The design was shaped by constraints, and several alternatives were compared on the same data. {T:alt} summarises the main choices."),
        ("h3", "4.3.1 Constraints"),
        ("bul", [
            "Compute: the laptop GPU has 4 GB of memory and about 55 minutes per epoch, so the classifier is trained on Google Colab and the backbones are the Tiny and Small sizes.",
            "Memory: Windows limits the commit memory, so no more than two data-loading workers are used and heavy jobs run one at a time.",
            "No physical hardware yet: sensors are simulated, the actuator is a log line and the latency on a Raspberry Pi is not measured.",
            "Data: training images are curated household and studio photographs; field footage differs from them (domain shift).",
            "Labels: some classes are genuinely ambiguous (cardboard boxes against packaging, steel against aluminium food cans).",
            "Time: one semester, which led to the staged plan and to software-first validation.",
        ]),
        ("h3", "4.3.2 Alternatives and Tradeoffs"),
        ("tbl", "alt", "Alternatives Considered and the Choices Made", ["Decision", "Alternatives", "Choice and tradeoff"], [
            ("Backbone", "ConvNeXt with ViT; ConvNeXt with Swin Transformer (Liu et al., 2021); frozen ConvNeXt with an RBF SVM", "Hybrid with ViT. The frozen-feature SVM scored 93.3% and exposed the fine-tuning problem; the improved recipe reached 93.83%. Swin trained for 2 epochs only (83.0%), so it shows only that the slot is swappable."),
            ("Class imbalance", "sampler; loss weights; both; logit adjustment", "One correction (sampler). Both together over-correct rare classes."),
            ("Uncertainty", "Monte Carlo dropout; maximum probability; deep ensemble", "Monte Carlo dropout inside a two-tier gate; the maximum probability is slightly better as an error detector (AUROC 0.66 against 0.60), so both are supported."),
            ("Hazard routing", "top-1 class; confidence gate; conformal family sets", "Conformal sets with a gate: a stated limit and 2.3 times lower leakage than a tuned confidence gate at about 8% review; the limit does not hold under heavy damage."),
            ("Missing sensor", "default value in the combined model; dedicated single-channel models", "Dedicated models: a calibrated score and a sensitivity target that holds without re-tuning."),
            ("Precision", "fp32; fp16 or int8 quantisation", "fp32 for analysis (fp16 flipped one of 4,232 predictions); quantisation is future work."),
            ("Learning", "central training; federated averaging", "Federated averaging matches central training in simulation and keeps images on the unit."),
        ], [2.6, 4.4, 7.5], {"font": 9}),
    ]
    return B
