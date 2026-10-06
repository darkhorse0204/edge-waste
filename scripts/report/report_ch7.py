# report_ch7.py - chapter 7 (cost analysis, results and discussion with all tables, figures and metric equations)
import pandas as pd
from pathlib import Path

from report_equations import *  # noqa: F401,F403

TITLE = ("CHAPTER 7", "COST ANALYSIS, RESULTS AND DISCUSSION")
ROOT = Path(__file__).resolve().parents[2]
s = S
E_ACC = wrap(N("Accuracy") + R(" = ") + F(R("1"), R("N")) + SUM(R("i=1"), R("N"), IND(HAT(s(R("y"), R("i"))) + R(" = ") + s(R("y"), R("i")))))
E_PRF = wrap(EQARR(
    s(N("Precision"), R("c")) + R(" = ") + F(s(R("TP"), R("c")), s(R("TP"), R("c")) + R(" + ") + s(R("FP"), R("c"))) + R(",   ") +
    s(N("Recall"), R("c")) + R(" = ") + F(s(R("TP"), R("c")), s(R("TP"), R("c")) + R(" + ") + s(R("FN"), R("c"))),
    s(R("F1"), R("c")) + R(" = ") + F(R("2 ") + s(N("Precision"), R("c")) + s(N("Recall"), R("c")), s(N("Precision"), R("c")) + R(" + ") + s(N("Recall"), R("c")))))
E_FAM = wrap(s(N("Accuracy"), N("family")) + R(" = ") + F(R("1"), R("N")) + SUM(R("i=1"), R("N"), IND(N("family") + D(HAT(s(R("y"), R("i")))) + R(" = ") + N("family") + D(s(R("y"), R("i"))))))
E_HR = wrap(s(N("Recall"), N("hazard")) + R(" = ") + F(s(R("TP"), R("H")), s(R("TP"), R("H")) + R(" + ") + s(R("FN"), R("H"))))
E_ECE = wrap(N("ECE") + R(" = ") + SUM(R("b=1"), R("B"), F(ABS(s(R("B"), R("b"))), R("N")) + ABS(N("acc") + D(s(R("B"), R("b"))) + R(" − ") + N("conf") + D(s(R("B"), R("b"))))))
E_LH = wrap(s(R("L"), R("H")) + R(" = ") + F(ABS(R("{ i : ") + s(R("y"), R("i")) + R(" ∈ H ") + T("and ") + R("a(") + s(R("x"), R("i")) + R(") = ") + T("recycling bin") + R(" }")), ABS(R("{ i : ") + s(R("y"), R("i")) + R(" ∈ H }"))))
E_W = wrap(R("W = ") + F(ABS(R("{ i : a(") + s(R("x"), R("i")) + R(") = ") + T("manual review") + R(" }")), R("N")))
E_FL = wrap(R("F = ") + F(R("1"), R("N")) + SUM(R("i"), "", IND(s(R("d"), N("drop")) + D(s(R("x"), R("i"))) + R(" ≠ ") + s(R("d"), N("full")) + D(s(R("x"), R("i"))))))


def baseline_rows():
    d = pd.read_csv(ROOT / "reports/ml_analysis/classical_baselines.csv")
    names = {"gaussian_naive_bayes": "Gaussian naive Bayes", "lda": "Linear discriminant analysis", "logistic_regression": "Logistic regression",
             "linear_svm": "Linear SVM", "rbf_svm": "RBF-kernel SVM", "knn": "k-nearest neighbours", "random_forest": "Random forest (300 trees)"}
    rows = []
    for m, nm in names.items():
        r = []
        for f in ("handcrafted", "imagenet_frozen", "finetuned_embedding"):
            x = d[(d.feature_set == f) & (d.model == m)].iloc[0]
            r.append(f"{x.test_accuracy * 100:.1f}% (train {x.train_accuracy * 100:.1f}%)")
        rows.append((nm, *r))
    rows.append(("Fine-tuned hybrid, own softmax head", "", "", "89.3% (train 96.9%)"))
    return rows


def blocks():
    B = [
        ("p", "This chapter reports the results, discusses what they mean and gives the cost analysis. Unless stated otherwise, results are for the first fine-tuned model (version 1), "
              "re-scored on its recovered and verified test split of 4,232 images, because that is the model that all the analysis, simulation and demonstration runs use. Results for the improved "
              "recipe (version 2) are those reported by the training platform on its own held-out split. Sensor results use simulated data and are labelled as such."),
        ("h2", "7.1 Evaluation Protocol and Metrics"),
        ("p", "The test set of 4,232 images (738 hazardous) was used once per model. Hyperparameters, thresholds and temperature were chosen on the validation set. Confidence intervals "
              "are 95% bootstrap intervals. The metrics follow Section 1.3.13. Accuracy over N items is defined in {E:acc}, where the hat marks the predicted class and the bracket is 1 when the two "
              "classes are equal. Precision, recall and F1 of class c follow {E:prf}, with true positives TP, false positives FP and false negatives FN; the macro F1 is the mean of F1<sub>c</sub> over classes. "
              "Family accuracy in {E:famacc} counts a prediction as right when the predicted family equals the true family. Hazard recall in {E:hr} uses the counts of the hazardous class group H. "
              "The expected calibration error in {E:ece} splits the predictions into B confidence bins B<sub>b</sub> and averages the gap between accuracy and mean confidence in each bin. Two operational "
              "metrics were defined for sorting. Hazard leakage L<sub>H</sub> in {E:lh} is the share of truly hazardous items that the routing action a(x<sub>i</sub>) sends to a recycling bin, and the review "
              "workload W in {E:w} is the share of all items sent for manual review. W is the price paid for a lower L<sub>H</sub>. When a sensor channel is lost, F in {E:fl} is the share of "
              "contamination decisions that change compared with the decision taken with all channels present."),
        ("eq", "acc", E_ACC), ("eq", "prf", E_PRF), ("eq", "famacc", E_FAM), ("eq", "hr", E_HR), ("eq", "ece", E_ECE), ("eq", "lh", E_LH), ("eq", "w", E_W), ("eq", "fl", E_FL),
        ("h2", "7.2 Classification Results"),
        ("p", "{T:head} gives the headline results. The first model reaches 89.30% item accuracy, 94.14% family (routing) accuracy and 93.50% hazard recall, with a macro F1 of 0.863 and a "
              "macro ROC-AUC of 0.981. The 4.8-point gap between item and family accuracy is the hierarchy working: the seven most frequent confusions are all inside one family "
              "({T:conf}). The weakest classes are exactly these look-alike pairs (F1 0.56 to 0.62), not the rare classes."),
        ("tbl", "head", "Headline Results on the Test Set of 4,232 Images", ["Metric", "First model (v1)", "95% bootstrap interval (v1)", "Improved recipe (v2)"], [
            ("Item accuracy (33 classes)", "89.30%", "88.42 to 90.19%", "93.83%"),
            ("Family (routing) accuracy", "94.14%", "93.43 to 94.80%", "97.28%"),
            ("Hazard recall", "93.50% (690 of 738)", "91.63 to 95.13%", "98.37%"),
            ("Hazard precision", "95.30%", "", ""),
            ("Balanced accuracy", "86.78%", "", ""),
            ("Macro F1", "0.863", "0.850 to 0.874", ""),
            ("Cohen's kappa; Matthews correlation", "0.885; 0.885", "", ""),
            ("Top-3; top-5 accuracy", "96.74%; 97.64%", "", ""),
            ("Macro ROC-AUC; macro PR-AUC", "0.981; 0.883", "", ""),
            ("Log loss", "0.651", "", ""),
        ], [4.4, 3.4, 3.7, 3.0], {"align_cols": ["l", "c", "c", "c"]}),
        ("tbl", "conf", "Most Frequent Confusions of the First Model", ["True class", "Predicted as", "Count", "Same family?"], [
            ("cardboard packaging", "cardboard boxes", "33", "yes"), ("cardboard boxes", "cardboard packaging", "21", "yes"),
            ("steel food cans", "aluminium food cans", "29", "yes"), ("aluminium food cans", "steel food cans", "18", "yes"),
            ("clothing", "shoes", "15", "yes (both textile)"), ("electronic waste", "medical waste", "12", "yes (both hazardous)"),
            ("medical waste", "electronic waste", "10", "yes (both hazardous)"),
        ], [4.0, 4.2, 2.0, 4.3], {"align_cols": ["l", "l", "c", "l"]}),
        ("fig", "pcpr", "pcpr", "Precision and Recall of Each Class on the Test Set (First Model)", 14.4),
        ("fig", "cmi", "cm_items", "Confusion Matrix at the Item Level (33 Classes, First Model)", 14.0, 18.0),
        ("fig", "cmf", "cm_fam", "Confusion Matrix at the Routing Level (Nine Families), Each Row Scaled to 100%", 11.0, 14.0),
        ("p", "The improved recipe of {T:recipe} lifts item accuracy by 4.5 points, family accuracy by 3.1 points and hazard recall by 4.9 points ({F:ver}). Validation accuracy was already 90.0% "
              "after the first epoch, in which only the new layers were trained, and 93.7% by epoch 5, against 89.5% after 15 epochs of the first recipe ({F:hist}); this supports the diagnosis that "
              "protecting the pretrained features is worth about 4.5 points. The validation loss was lowest at epoch 5 and rose slowly afterwards while the training loss kept falling, a mild "
              "overfitting trend after epoch 8, which is why the best epoch was 8 of 12. The improved figures are those reported by the training platform: re-scoring the checkpoint on a "
              "locally rebuilt split gave 96.8%, which is higher than the platform's own 93.8% and is the signature of leakage (the local and platform splits contain different images), so the "
              "local figure was discarded and the evaluation command now refuses a split that does not reproduce the validation accuracy recorded in the checkpoint. Re-scoring the improved "
              "model on the exact platform split is listed as an open item."),
        ("fig", "ver", "versions", "Item Accuracy, Family Accuracy and Hazard Recall of the Two Hybrid Recipes and of the Frozen-Feature SVM", 12.0),
        ("fig", "hist", "v2hist", "Training History of the Improved Recipe: Frozen-Backbone Epochs Are Shaded", 14.2),
        ("h2", "7.3 Fit Diagnostics: Overfitting and Underfitting"),
        ("p", "{T:fit} compares training, validation and test accuracy of the first model in evaluation mode without augmentation. The verdict is a moderate generalisation gap, not harmful "
              "overfitting, and not underfitting. The model is not underfitting because it reaches 96.9% training accuracy. It has some variance, with a 7.6-point gap between training and "
              "test accuracy, which is expected for 50.7 million parameters and 19.7 thousand images. The overfitting is not harmful because the best validation accuracy came at the last epoch, "
              "so the run was stopped by the epoch budget and not by early stopping, and validation and test accuracy differ by only 0.21 points, so the choices made on validation transfer to unseen "
              "data. The largest per-class gaps are the look-alike pairs (cardboard packaging 82.9% training against 50.7% test), and even their training accuracy is low, which points to "
              "ambiguous labels rather than memorisation. The learning curve of a logistic regression on frozen features rises from 84.5% validation accuracy at 394 images to 92.3% at 13.8 "
              "thousand and is flat from there to 19.7 thousand, so more data of the same kind gives diminishing returns. The validation curve shows the textbook picture: the same model underfits "
              "at a regularisation constant C of 10<sup>−4</sup> (88.3% training, 87.6% validation), is best at C = 0.01 (96.0% and 92.4%) and overfits beyond (98.9% and 90.9% at C = 100)."),
        ("tbl", "fit", "Accuracy and Loss on the Three Splits (First Model)", ["Split", "Images", "Accuracy", "Log loss"], [
            ("Training", "19,739", "96.89%", "0.379"), ("Validation", "4,232", "89.51%", "0.653"), ("Test", "4,232", "89.30%", "0.651"),
        ], [4.0, 3.0, 3.5, 3.5], {"align_cols": ["l", "c", "c", "c"]}),
        ("fig", "fitd", "fit", "Fit Diagnostics: Learning Curve, Validation Curve and Per-Class Train-to-Test Gaps", 14.4),
        ("p", "The measures taken against overfitting were ImageNet pretraining, data augmentation, dropout, weight decay, label smoothing, early stopping on validation and a test set scored once."),
        ("h2", "7.4 Calibration"),
        ("p", "The first model is under-confident: it is right 89% of the time but on average only 73% sure, the known side effect of label smoothing and class-balanced training, "
              "which pull probabilities towards uniform. Temperature scaling with T = 0.71 fitted on the validation set halves the expected calibration error from 16.56% to 7.30% "
              "without changing a single prediction ({T:cal}, {F:rel}). Under-confidence matters downstream because the uncertainty gate sends low-confidence items to a person."),
        ("tbl", "cal", "Calibration Before and After Temperature Scaling", ["Test", "ECE", "Max CE", "NLL", "Brier", "Mean confidence", "Accuracy"], [
            ("As trained", "16.56%", "45.1%", "0.651", "0.232", "73.0%", "89.30%"),
            ("Temperature-scaled (T = 0.71)", "7.30%", "32.2%", "0.534", "0.183", "89.7%", "89.30%"),
        ], [4.2, 1.6, 1.7, 1.5, 1.5, 2.3, 1.8], {"align_cols": ["l", "c", "c", "c", "c", "c", "c"], "font": 9}),
        ("fig", "rel", "rel", "Reliability Diagram Before and After Temperature Scaling (First Model)", 14.2),
        ("h2", "7.5 Class Imbalance"),
        ("p", "Twenty-seven classes have 350 training images each, and six merged classes are larger, so the ratio of the largest to the smallest class is 11.6 ({F:imb}). The first model used "
              "a balanced sampler and inverse-frequency loss weights together. The analysis found that either mechanism alone already makes training behave as if all classes were equally common, "
              "so together the rare classes are weighted about 1/n² instead of 1/n and the posteriors shift towards them; office paper, for example, was predicted 1.32 times as often as it occurs. "
              "Without retraining, the post-hoc logit adjustment of {E:logadj} with τ = 1.75, chosen on validation, raises accuracy from 89.30% to 89.60%, macro F1 from 0.863 to 0.870 and "
              "hazard recall from 93.50% to 94.72%, while balanced accuracy moves from 0.868 to 0.862 ({T:imb}). The effect is real but under one point of accuracy, because the imbalance mainly "
              "involves six large classes. The training configuration now applies one correction only."),
        ("fig", "imb", "imbalance", "Class Imbalance: Training Images per Class", 14.2),
        ("tbl", "imb", "Effect of Post-Hoc Logit Adjustment on the Test Set", ["Setting", "Accuracy", "Macro F1", "Hazard recall", "Balanced accuracy"], [
            ("As trained", "89.30%", "0.863", "93.50%", "0.868"),
            ("τ = 1 (about one correction)", "89.86%", "", "94.85%", ""),
            ("τ = 1.75 (chosen on validation)", "89.60%", "0.870", "94.72%", "0.862"),
        ], [5.0, 2.4, 2.2, 2.6, 2.8], {"align_cols": ["l", "c", "c", "c", "c"]}),
        ("h2", "7.6 Classical Baselines: PCA, LDA, SVM and Others"),
        ("p", "To test whether the deep model earns its complexity, seven classical classifiers were trained on three feature sets with the same protocol as the deep model: standardise, reduce with "
              "principal component analysis (PCA), classify (with the support-vector machine of Cortes & Vapnik, 1995, and the other models of the scikit-learn library; Pedregosa et al., 2011), choose hyperparameters on validation and score the test set once. The feature sets were hand-crafted features (a colour "
              "histogram plus histograms of oriented gradients, 1,860 values), frozen ImageNet ConvNeXt features without fine-tuning (768 values), and the fine-tuned hybrid's own embedding "
              "(1,024 values). {T:base} and {F:base} give the test accuracies."),
        ("tbl", "base", "Test Accuracy of Seven Classical Classifiers on Three Feature Sets", ["Classifier", "Hand-crafted", "Frozen ImageNet", "Fine-tuned embedding"], baseline_rows(), [4.6, 3.4, 3.4, 3.6], {"font": 9, "align_cols": ["l", "c", "c", "c"]}),
        ("fig", "base", "baselines", "Test Accuracy of the Classical Classifiers on Three Feature Sets", 14.4),
        ("p", "Four conclusions follow. First, deep features are needed: the best classical computer-vision pipeline (colour and gradient features, PCA, RBF SVM) stops at 71.1%, while deep features "
              "reach 89% to 93% (McNemar paired test p ≈ 5 × 10<sup>−135</sup>). Second, the table shows bias and variance side by side: naive Bayes and LDA have the smallest train-to-test gaps but the lowest "
              "accuracy (high bias), random forest and kNN reach 99.4% training accuracy but lose 8 to 42 points on test (high variance), and the RBF SVM with C chosen on validation sits in between. "
              "Third, and most important, an RBF SVM on the frozen, un-fine-tuned ImageNet features (93.3% item, 97.1% family and 98.1% hazard recall) beat the fine-tuned hybrid on every metric "
              "(p ≈ 8 × 10<sup>−21</sup>), which exposed the fine-tuning distortion and led to the improved recipe. Fourth, every classifier on the fine-tuned embedding lands between 88.5% and 89.8%, next to the "
              "network's own head (89.3%), so the head is not the bottleneck. The PCA analysis ({F:pca}) shows that the fine-tuned embedding packs 95% of its variance into 29 dimensions, against 715 for "
              "the hand-crafted features and 513 for the frozen features. This is close to the 32 (classes minus one) that the neural-collapse picture predicts for a trained last layer."),
        ("fig", "pca", "pca", "Cumulative Explained Variance of Principal Components for the Three Feature Sets", 11.5),
        ("h2", "7.7 What the Model Learned"),
        ("p", "{F:emb} shows PCA, LDA and t-SNE (van der Maaten & Hinton, 2008) projections of the test set coloured by family. The t-SNE view of frozen ImageNet features already shows loose family clusters; after "
              "fine-tuning the clusters are compact and mostly separated, with overlap where the confusion matrix says it should be (cardboard, metal cans, electronic and medical waste). "
              "The fusion layer gives the ConvNeXt stream 82% to 93% of the attention weight for every family (textile 93%, cardboard 92%, paper 82%) and the Vision Transformer 7% to 18% "
              "({F:attn}). The model therefore relies mainly on the convolutional stream, because texture carries most of the signal for material recognition. The attention weight is an "
              "indicator and not an exact attribution, because the two projected features can differ in magnitude. This suggests a ConvNeXt-only model as the next ablation."),
        ("fig", "emb", "embed", "PCA, LDA and t-SNE Projections of the Test Set, and t-SNE of Frozen Features Against the Fine-Tuned Embedding", 14.4),
        ("fig", "attn", "attn", "Attention Weight Given to the Two Backbones for Each Family", 12.0),
        ("h2", "7.8 Uncertainty, Hazard Gating and Conformal Routing"),
        ("h3", "7.8.1 Two-Tier Gate on Out-of-Distribution Footage"),
        ("p", "Running the real classifier on out-of-distribution video crops (outdoor litter) showed that an uncertain classifier disproportionately guesses hazard classes on unfamiliar input. "
              "On a 240-frame test clip, hazard predictions without the uncertainty gate would have fired 12 automatic hazardous-stream actuations for items that were physically cardboard and "
              "cups. With the two-tier gate switched on, automatic hazardous actions fell to zero: all 12 items went to priority manual review and none reached ordinary recycling. The mean uncertainty on "
              "such crops was 0.68, against near zero on in-distribution images."),
        ("h3", "7.8.2 Uncertainty Estimators Compared"),
        ("p", "On the 4,232 held-out items with 25 stochastic passes, the average Monte Carlo dropout uncertainty is higher for wrong predictions than for right ones (0.463 against 0.394). "
              "As a detector of misclassified items, however, it is not better than the plain maximum class probability: the area under the ROC curve is 0.598 for Monte Carlo dropout and 0.657 "
              "for the maximum probability (0.660 and 0.692 at routing level), and accepting the 50% most certain items gives 90.9% against 93.2% item accuracy ({F:unc}). "
              "A fixed threshold of 0.5 sends 34.2% of all items to review, because label-smoothed training raises the entropy of correct predictions. The conclusion is that the "
              "benefit of the gate comes from its two-tier structure and from a threshold calibrated to a review budget ({E:tau}), and both uncertainty measures work inside it."),
        ("fig", "unc", "unc", "Uncertainty Measures on Held-Out Data: Distributions, and Accuracy of Accepted Items as the Accepted Share Grows", 14.4),
        ("h3", "7.8.3 Conformal Family Sets"),
        ("p", "Top-1 routing with no gate sends 48 of the 738 hazardous test items (6.5%) to a recycling bin. The conformal family-set router was evaluated by halving the test split at random "
              "300 times for each setting: one half sets the per-family thresholds and the other half is routed and scored. The error level for the non-hazardous families is α = 0.10 and "
              "α<sub>H</sub> is varied. The comparison is a maximum-probability gate whose threshold is tuned on the calibration half to give the same review workload, which is a fair test because any gate can lower "
              "leakage by reviewing more items. {T:cp} and {F:cc} give the result."),
        ("tbl", "cp", "Hazard Leakage of Conformal Family Sets Against a Matched Confidence Gate", ["α<sub>H</sub>", "Conformal leakage (mean / 95th percentile)", "Items to review W", "Matched confidence gate (mean / 95th percentile)", "Advantage"], [
            ("0.05", "2.84% / 4.26%", "5.7%", "3.50% / 4.58%", "1.2 times"),
            ("0.02", "1.57% / 2.90%", "7.8%", "3.55% / 5.09%", "2.3 times"),
            ("0.01", "0.78% / 2.05%", "31.1%", "1.77% / 4.52%", "2.3 times"),
            ("0.005", "0.29% / 1.11%", "39.5%", "1.37% / 2.17%", "4.7 times"),
        ], [1.9, 3.8, 2.4, 4.0, 2.4], {"align_cols": ["c", "c", "c", "c", "c"], "font": 9.5}),
        ("fig", "cc", "conf_curve", "Hazard Leakage Against Review Workload for Conformal Family Sets and a Matched Confidence Gate", 14.4),
        ("p", "For every α<sub>H</sub> of 0.05 or less, the measured mean leakage stays below α<sub>H</sub>. At practical workloads (about 8% review) the advantage over a tuned confidence gate is 2.3 times, and "
              "the distinguishing property of the mechanism is the stated bound. In a simulated check where the true coverage can be computed exactly (n = 369 calibration hazards, 20,000 draws), "
              "the Beta-corrected rank ({E:brank}) missed the target in 2.1% of calibrations, within the chosen δ of 5%, while the standard rank missed in 39.5%. On real data the Beta-corrected version "
              "is very careful: about 11% of items are routed automatically at α<sub>H</sub> = 0.02 and δ = 0.05, so it is the conservative choice."),
        ("h3", "7.8.4 Routing Outcomes as Each Mechanism Is Added"),
        ("p", "{T:abl} shows what happens to the held-out items as each mechanism is added, using 200 random splits, α = 0.10, α<sub>H</sub> = 0.05 and gates set to a 10% review budget. Conformal "
              "sets alone reach 2.88% leakage while reviewing 5.6% of items. A confidence gate needs to review 10.1% to reach 3.30%, and the Monte Carlo dropout gate 10.1% for 4.88%. Combining "
              "family sets with a gate gives the lowest leakage (2.27% with the maximum-probability gate) at a review workload of 12.4% ({F:abl}). The cost of every mechanism is a lower share of items "
              "placed in a correct bin automatically (94.1% for top-1 down to 84.6%), because the items that are sent to a person are not counted as correct bins; this is the price of safety."),
        ("tbl", "abl", "Routing Results as Each Mechanism Is Added", ["Configuration", "Review share W", "Hazard leakage", "Non-hazard sent to hazard bin", "Correct bin overall"], [
            ("top-1 class only", "0.0%", "6.60%", "0.96%", "94.1%"), ("maximum-probability gate", "10.1%", "3.30%", "0.68%", "86.1%"),
            ("Monte Carlo dropout gate", "10.1%", "4.88%", "0.84%", "85.3%"), ("conformal family sets", "5.6%", "2.88%", "1.14%", "90.3%"),
            ("conformal sets + Monte Carlo dropout gate", "13.0%", "2.64%", "0.85%", "83.5%"), ("conformal sets + maximum-probability gate", "12.4%", "2.27%", "0.66%", "84.6%"),
        ], [5.4, 2.4, 2.3, 2.7, 2.2], {"align_cols": ["l", "c", "c", "c", "c"], "font": 9.5}),
        ("fig", "abl", "ablation", "Where Each Held-Out Item Ends Up Under Each Routing Set-Up, and Hazard Leakage With Its Review Cost", 14.4),
        ("h2", "7.9 Robustness to Damaged Images"),
        ("p", "The conformal guarantee is about items that resemble the calibration items. To see what happens when they do not, 1,500 held-out test images were damaged at three strengths by "
              "each of four kinds of damage (Gaussian noise, Gaussian blur, darkening, and a black square covering part of the image), while the thresholds were calibrated only on 2,732 clean "
              "images that were not among the tested ones. {T:shift} and {F:shift} give the results. Item accuracy falls from 89.7% on clean images to 53.9% at noise σ = 0.2 and 15.3% at blur σ = 4, "
              "and these tests are deliberately hard. The family sets alone no longer keep the 5% limit under heavy damage (15.6% leakage at noise σ = 0.2), but adding an uncertainty gate lowers the risk "
              "further (9.8% with the maximum-probability gate and 10.9% with the Monte Carlo dropout gate, against 28.5% for top-1 routing). With moderate damage (item accuracy about 77% or better) the "
              "combined mechanism keeps leakage at or close to the 5% target; with heavy damage it still cuts leakage to about one third of top-1 routing or less. The maximum-probability gate flags more "
              "of the damaged images than the Monte Carlo dropout gate and gives equal or lower leakage in every condition tested. The stated limit therefore applies to items similar to the "
              "calibration items, and the gate gives added protection beyond that."),
        ("tbl", "shift", "Hazard Leakage and Non-Hazard Items Sent to the Hazardous Bin Under Damaged Images (%)", ["Condition", "Item accuracy", "Hazard leakage: top-1 / sets / sets + MC gate / sets + max-prob. gate", "Non-hazard to hazard bin: same four set-ups"], [
            ("clean images", "89.7%", "5.9 / 2.3 / 2.3 / 1.6", "0.7 / 1.3 / 0.7 / 0.5"), ("noise σ = 0.1", "76.9%", "16.4 / 7.0 / 5.9 / 5.9", "3.9 / 4.3 / 2.5 / 1.8"),
            ("noise σ = 0.2", "53.9%", "28.5 / 15.6 / 10.9 / 9.8", "8.1 / 9.0 / 3.9 / 3.1"), ("blur σ = 2", "77.2%", "3.1 / 1.2 / 1.2 / 1.2", "13.3 / 13.9 / 10.9 / 8.4"),
            ("blur σ = 4", "15.3%", "47.3 / 7.8 / 7.8 / 7.8", "32.3 / 40.4 / 17.7 / 12.5"), ("darkness × 0.15", "71.6%", "8.6 / 2.3 / 2.3 / 2.3", "12.9 / 14.5 / 9.7 / 7.1"),
            ("half of the image covered", "64.9%", "34.8 / 13.7 / 10.9 / 9.0", "3.9 / 4.2 / 2.3 / 1.4"),
        ], [3.5, 2.0, 4.6, 4.4], {"align_cols": ["l", "c", "c", "c"], "font": 9}),
        ("fig", "shift", "shift", "Behaviour Under Damaged Images: Item Accuracy, Share Flagged, Hazard Leakage and Automatic Routing", 14.4),
        ("h2", "7.10 Contamination Index and the Sensor-Loss Study"),
        ("p", "The index was checked from start to finish on simulated calibration data that follows the planned physical procedure (two item categories, five residue levels, 300 repeats): "
              "scaling with fixed anchors, three fitted models, five-fold cross-validated comparison, a threshold that meets the sensitivity target, and all three sensor-availability paths. "
              "On a held-out draw the combined model has an area under the ROC curve of 0.81 and each single-channel model 0.76 ({F:roc}). SHAP shows that each sensor's contribution to the "
              "output adds up exactly, to within 10<sup>−6</sup>. The synthetic sensors are only weakly separable, so a threshold that reaches 95% sensitivity gives a high false-positive rate "
              "(84% in the first small test); real calibration data are essential, which is why the physical sensors are the next stage."),
        ("fig", "roc", "oci_roc", "Receiver Operating Characteristic Curves of the Three Contamination Models on Simulated Sensors", 9.5),
        ("p", "The second simulation tests the three-model design: what happens when one channel is lost, using 3,000 training and 3,000 separate test samples at three noise levels? Because the "
              "models are linear, replacing a missing input by a constant leaves the ranking unchanged, so the area under the curve is the same (0.761 with moisture only at noise × 1.0) and, if the "
              "threshold is tuned again, the two approaches make the same decisions. The benefit of the dedicated models is that they need no extra tuning. With the combined model's threshold left in "
              "place, a default value of zero for the lost gas channel gives 53.7% sensitivity, and for a lost moisture channel 14.9%, against the 95% target. Using the training average as the default "
              "gives a very high false-positive rate (94.4% and 99.3%). Each dedicated model with its own threshold meets the target (94.9% and 94.5%), and its score is better calibrated "
              "(Brier score 0.194 against 0.272 for default zero). Out of every 100 items, 8.6 decisions change with the dedicated moisture-only model against 48.5 with the default value ({E:fl}). "
              "The benefit is therefore a calibrated score and a dependable sensitivity target in every case; with physical sensors the models may differ more, for example if an interaction term "
              "makes them non-linear."),
        ("tbl", "oci", "Contamination Index With One Lost Sensor Channel (Simulated Sensors, Noise × 1.0)", ["Case and method", "Sensitivity", "False positive rate", "Brier score"], [
            ("gas lost: dedicated model with its own threshold", "94.9%", "81.6%", "0.194"), ("gas lost: default 0, shared threshold", "53.7%", "16.6%", "0.272"),
            ("gas lost: training average as default, shared threshold", "98.5%", "94.4%", "0.199"), ("gas lost: default 0, threshold tuned again", "94.9%", "81.6%", "0.272"),
            ("moisture lost: dedicated model with its own threshold", "94.5%", "83.4%", "0.198"), ("moisture lost: default 0, shared threshold", "14.9%", "0.7%", "0.399"),
            ("moisture lost: training average as default, shared threshold", "99.7%", "99.3%", "0.203"), ("moisture lost: default 0, threshold tuned again", "94.5%", "83.4%", "0.399"),
        ], [7.0, 2.4, 2.8, 2.3], {"align_cols": ["l", "c", "c", "c"], "font": 9.5}),
        ("fig", "od", "oci_drop", "Sensitivity, False Positive Rate and Brier Score With One Channel Lost (Simulated Sensors)", 14.4),
        ("h2", "7.11 Federated Learning"),
        ("p", "Federated averaging ({E:fed}) was simulated on the stored 1,024-value embeddings with five sorting units that each see a different mix of classes (a Dirichlet split with parameter 0.3, "
              "giving 6,003, 3,772, 4,621, 2,378 and 2,965 training images). In this simulation the feature extractor is kept fixed and the classifier layers are averaged for 15 rounds with 2 "
              "local epochs each. The shared model reaches 89.4% accuracy on the 4,232 test images, which matches the 89.0% of one central model trained on all images and is well above the "
              "81.4% average of units that train alone (from 70.1% to 87.8%) ({F:fedr}). Only parameters were exchanged. A federated test across several physical devices, and with the "
              "full backbones, is the next stage."),
        ("fig", "fedr", "federated", "Federated Averaging With Five Units Against One Central Model and Units Training Alone", 11.5),
        ("h2", "7.12 Detection, Model Cost and Latency"),
        ("p", "The YOLO26n detector (2.4 million parameters) was fine-tuned on TACO at 640 pixels on the laptop GPU ({T:detr}). Naming the object as well as finding it costs 0.32 in mAP at an overlap of "
              "0.50. The detector is strong on rigid objects (bottle cap 0.64, cup 0.61, can 0.59) and weak on tiny or amorphous ones (pop tab 0.08, broken glass 0.06), and takes 4.2 ms per image on the GPU."),
        ("tbl", "detr", "Detector Results on the 900-Image Validation Split", ["Variant", "Precision", "Recall", "mAP at IoU 0.50", "mAP at IoU 0.50 to 0.95"], [
            ("1-class waste item locator", "0.871", "0.620", "0.700", "0.492"), ("18-class object identifier", "0.603", "0.361", "0.383", "0.289"),
        ], [5.0, 2.2, 2.2, 2.6, 2.6], {"align_cols": ["l", "c", "c", "c", "c"]}),
        ("fig", "dpr", "det_pr", "Precision-Recall Curve of the 18-Class Detector on the Validation Split", 10.0),
        ("fig", "dcm", "det_cm", "Normalised Confusion Matrix of the 18-Class Detector", 11.5, 13.0),
        ("p", "{T:cost} gives the cost of the classifier, measured with batch size 1 at 224 × 224 pixels. The 50.75 million parameters need 17.4 GFLOPs per image, which is 14.0 ms on the laptop GPU "
              "and 115.5 ms on the laptop CPU; both are inside the 500 ms target. Latency on a Raspberry Pi 5 has not been measured. The ONNX files (203 MB for the classifier and 9.3 MB for the "
              "detector, 32-bit) match the original models to within 9.4 × 10<sup>−6</sup>, and 16-bit or 8-bit quantisation, which would roughly halve or quarter the size, is future work."),
        ("tbl", "cost", "Model Cost of the Classifier and the Detector", ["Quantity", "Value"], [
            ("Parameters (ConvNeXt-Tiny / ViT-Small / projections / fusion / head)", "27.82 M / 21.67 M / 0.59 M / 0.13 M / 0.54 M; total 50.75 M"),
            ("Compute per image", "17.4 GFLOPs (8.7 G multiply-adds)"), ("Latency, laptop GPU (RTX 2050)", "14.0 ms"), ("Latency, laptop CPU", "115.5 ms"),
            ("Checkpoint and ONNX file, classifier (32-bit)", "203 MB"), ("Detector (YOLO26n)", "2.4 M parameters; 4.2 ms; 9.3 MB ONNX"),
        ], [7.0, 7.5], {"font": 9.5}),
        ("h2", "7.13 Explanations and Video Survey"),
        ("p", "Grad-CAM and LIME agreed on the same test images (Chapter 6), and SHAP contributions add up to the contamination score with an error below 10<sup>−6</sup>. The MobileSAM outline "
              "removed the background from the classifier input (in the first tested case 36% of the detector box was background). The video survey merged 160 per-frame detections into 16 "
              "inventory entries. An earlier project note had stated that 19 items were tracked in this clip and that this matched the true count; this could not be reproduced (re-runs gave 13 to 17 "
              "items, depending on the detector) and the clip has no ground-truth count, so it is reported here only as a de-duplication result."),
        ("h2", "7.14 Cost Analysis"),
        ("h3", "7.14.1 Development Cost"),
        ("p", "The software stage was built with open-source tools and public datasets, so there is no licence cost. {T:devc} lists the resources used. The classifier is trained in about one "
              "GPU-hour per run (48 minutes for the improved recipe), so the cost of retraining on a cloud GPU is small."),
        ("tbl", "devc", "Development Resources and Their Cost", ["Resource", "Use", "Cost"], [
            ("Python, PyTorch, timm, Ultralytics, scikit-learn and other libraries", "all software", "open source, no licence fee"),
            ("Kaggle and TACO datasets", "training and evaluation data", "free for research use"),
            ("Google Colab GPU", "classifier training, about 1 hour per run", "depends on the plan used; small"),
            ("Laptop with RTX 2050 GPU", "detector training, analysis, demonstrations", "already owned"),
            ("Team effort", "three students, one semester", "academic project"),
        ], [6.2, 5.0, 3.3], {"font": 9.5}),
        ("h3", "7.14.2 Estimated Cost of the Hardware Prototype"),
        ("p", "{T:bom} gives the estimated bill of materials of one prototype unit, which {F:cost} shows as a chart. The prices are approximate planning estimates in Indian rupees, with an uncertainty of "
              "about 30%, and must be confirmed with vendor quotations before purchase. The total is about 21,000 rupees, dominated by the single-board computer, the conveyor mechanism and the "
              "camera. The sensors, which are the novel part, cost under 1,000 rupees together, which supports the claim that the contamination index adds information at very low cost."),
        ("tbl", "bom", "Estimated Bill of Materials of One Prototype Unit (Estimate)", ["Part", "Estimated cost (rupees)"], [
            ("Edge computer (Raspberry Pi 5 class)", "about 9,000"), ("Camera module", "about 3,000"), ("Moisture and gas sensors", "about 600"),
            ("Analogue-to-digital converter", "about 400"), ("Servos and motor driver", "about 1,500"), ("Conveyor belt mechanism and frame", "about 4,000"),
            ("Bins, wiring and power supply", "about 2,500"), ("Total", "about 21,000"),
        ], [8.0, 5.0], {"align_cols": ["l", "c"]}),
        ("fig", "cost", "cost", "Estimated Cost of the Parts of One Prototype Unit (Estimate to Be Confirmed With Vendor Quotations)", 11.5),
        ("h3", "7.14.3 Operating Cost and Benefit"),
        ("p", "Running cost is expected to be small: a unit of this class should draw only a few watts and no cloud service is needed, because inference runs on the device and only model parameters are exchanged in federated rounds. "
              "The benefit is not measured in this report and is stated only as a design expectation: a lower share of hazardous items reaching recycling, fewer contaminated items passing, and one inventory "
              "entry per item in surveys. Its size can only be measured on a real conveyor."),
        ("h2", "7.15 Discussion"),
        ("p", "The results support the design. The hybrid classifier recognises 33 classes with 89.3% to 93.8% accuracy, and the routing view shows that most confusions do not change the bin. "
              "The calibrated family sets give a stated, checkable hazard limit, with 2.3 times lower leakage than a tuned confidence gate at a similar workload. The contamination index design keeps its "
              "sensitivity target when a sensor is lost, with no manual re-tuning. The explanation, segmentation, federated and video components work together on the same models."),
        ("p", "Several findings were unfavourable, and they are reported as found because they shaped the design. The fine-tuning recipe of the first model was sub-optimal, as the frozen-feature SVM "
              "showed. Monte Carlo dropout is not a better error detector than the maximum class probability, and its fixed threshold of 0.5 over-flags. The conformal limit does not survive heavy image "
              "damage, so the uncertainty gate is kept as a second line of defence. For linear models, a dedicated single-channel model gives the same decisions as imputation with a re-tuned threshold, so "
              "its benefit is calibration and a standing sensitivity target, not better ranking. Two data problems (a path-hash leak and stale duplicate files) inflated accuracy before they were found, "
              "which is why the split guard now exists."),
        ("p", "The main limitations are the following. There is no physical hardware yet, so the contamination model is fitted on simulated data that are only weakly separable and latency on the "
              "Raspberry Pi is unmeasured. The training photographs are studio and curated household images, so on outdoor litter the classifier is unsure and the system sends most items to a person, which is "
              "safe but shows that site images and recalibration are needed. The improved model's figures rest on the training platform and still have to be reproduced locally. The federated study is a "
              "single-process simulation on fixed features. These limitations define the work of the next stage, and none of them changes the conclusions about the design mechanisms."),
    ]
    return B
