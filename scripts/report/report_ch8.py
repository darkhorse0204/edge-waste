# report_ch8.py - chapter 8 (summary of findings, conclusions, recommendations and future work)
TITLE = ("CHAPTER 8", "SUMMARY")


def blocks():
    return [
        ("p", "This project set out to build an edge-deployable waste-sorting system that is accurate, safe, honest about its uncertainty and able to learn across units without sharing "
              "images. The work was done in software on real held-out data, with sensors simulated and the physical prototype planned as the next stage."),
        ("h2", "8.1 Summary of the Work"),
        ("p", "Three public datasets were consolidated into a taxonomy of 33 item classes in nine material families, with a leakage-checked 70 : 15 : 15 split. A hybrid classifier "
              "joins a ConvNeXt stream and a Vision Transformer stream with learned attention fusion, and a YOLO detector supplies the object name for a hazard-exempt cross-check. "
              "Uncertainty is estimated by Monte Carlo dropout and temperature scaling, and routing follows calibrated sets of material families with a stricter limit for hazards. "
              "An Organic Contamination Index from a moisture sensor and a gas sensor, with three models chosen by sensor availability, scores contamination that a camera cannot see. "
              "Grad-CAM, LIME, SHAP and the Segment Anything Model explain and refine decisions, federated averaging lets units learn together without sharing images, and a tracker gives a "
              "de-duplicated inventory from video."),
        ("h2", "8.2 Findings"),
        ("bul", [
            "The first model reached 89.30% item accuracy, 94.14% family (routing) accuracy and 93.50% hazard recall on 4,232 held-out images (macro F1 0.863). The improved recipe, "
            "which protects the pretrained features by training the new layers first, reached 93.83%, 97.28% and 98.37% as reported by the training platform.",
            "Seven of the eight most common confusions are between look-alike items that share a bin, so reporting recognition and routing separately explains the gap between them.",
            "An SVM on frozen ImageNet features beat the first fine-tuned model, which exposed fine-tuning distortion; deep features beat the best hand-crafted pipeline by about 20 points.",
            "The first model is under-confident, and one temperature (0.71) halves the expected calibration error from 16.6% to 7.3% without changing predictions.",
            "Class imbalance had been corrected twice; correcting it once, or adjusting the logits afterwards, changes accuracy by less than one point.",
            "Conformal family sets cut hazard leakage from 6.5% to 1.57% at 7.8% of items sent for review, 2.3 times lower than a tuned confidence gate, with a stated and checkable limit for items like the calibration set.",
            "Under heavy image damage the limit no longer holds, but the uncertainty gate cuts leakage to about one third of top-1 routing or less.",
            "On a test clip of unfamiliar items the two-tier gate reduced false hazardous actuations from 12 to 0.",
            "On simulated sensors, dedicated single-channel models keep a sensitivity of 94.9% and 94.5% with one channel lost, where a default value in the combined model gives 53.7% and 14.9%.",
            "Federated averaging over five units with different class mixes matched a central model (89.4% against 89.0%) while exchanging only parameters.",
        ]),
        ("h2", "8.3 Conclusion"),
        ("p", "The project shows that putting safety and calibration around a strong classifier gives a sorter that can say how sure it is, keeps hazardous items out of recycling with a stated "
              "limit, uses cheap sensors to find contamination that the camera cannot see, and explains its decisions. The honest reporting of unfavourable results (an over-flagging uncertainty "
              "threshold, a limit that weakens under damage, and two data leaks that were found and fixed) is part of the result, because each of them shaped a design choice. The system is at "
              "Technology Readiness Level 3: an experimental proof of concept with the software validated on real held-out data and the hardware still to be built."),
        ("h2", "8.4 Recommendations"),
        ("num", [
            "Build the physical prototype and collect real calibration data for the contamination index before relying on any contamination figure.",
            "Collect images at the target site, fine-tune on them and recalibrate the conformal thresholds, because the training photographs are studio and household images.",
            "Prefer a threshold calibrated to a review budget over the fixed uncertainty threshold of 0.5, and consider the maximum-probability gate beside Monte Carlo dropout.",
            "Re-score the improved model on the exact training-platform split to confirm the reported figures, and report them with confidence intervals.",
            "Quantise the models and measure latency on the target board before choosing the final backbone sizes.",
        ]),
        ("h2", "8.5 Future Work"),
        ("bul", [
            "Run the physical experiment: Raspberry Pi 5 deployment, sensor calibration with real residues, conveyor and diverter, and a federated round across at least two physical units.",
            "Test a ConvNeXt-only model and larger backbones, since the fusion layer gives most of its weight to the convolutional stream.",
            "Add field and plant datasets such as WaRP for a robustness evaluation, and fine-tune the detector on the hazardous classes, which it currently cannot name.",
            "Improve the conformal limit under shift, for example with a recalibration that uses a small labelled sample from the new site.",
            "Measure the quality of the generated images with a standard score and test whether they help rare classes.",
            "Study the benefit on a real conveyor: contaminated items caught, hazardous items kept out of recycling and operator workload.",
        ]),
    ]
