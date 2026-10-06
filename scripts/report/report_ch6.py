# report_ch6.py - chapter 6 (project demonstration: real images, video, explanations and simulated sensors)
TITLE = ("CHAPTER 6", "PROJECT DEMONSTRATION")


def blocks():
    B = [
        ("p", "This chapter shows the working system. Every figure was produced by a script in the repository from the stored classifier outputs and the saved demonstration runs, "
              "on images that were never used for training. Images were chosen by a fixed random seed and not by hand, so they show typical behaviour, including the occasional "
              "wrong top class. Sensor figures are simulated and are labelled so."),
        ("h2", "6.1 Demonstration Set-Up"),
        ("p", "The demonstrations run on the laptop in {T:dev} with the commands in {T:demo}. The live camera pipeline detects, classifies and routes items from a webcam; the "
              "other commands write figures and videos under the reports folder, which are the source of the figures in this chapter. The set-up for the routing demonstrations is the same "
              "as in the evaluation: thresholds are calibrated on one random half of the held-out images and the demonstration images come from the other half, with a hazard level "
              "α<sub>H</sub> of 0.05, α of 0.10 and a review budget β of 10%."),
        ("tbl", "demo", "Demonstration Commands and Outputs", ["Demonstration", "Command", "Output"], [
            ("Live detect, classify and route", "python scripts/run_live_camera.py", "webcam window with boxes, labels and a heat map"),
            ("Routing decisions on held-out images", "python scripts/make_demo_snapshots.py", "routing_demo.png, explain_panel.png"),
            ("All 33 classes, hazards, heat maps, detector, outlines, videos", "python scripts/make_demo_gallery.py classes (and hazards, gradcam, detector, videos, oci, federated, gan)", "reports/demo/gallery/"),
            ("Video litter survey", "edgewaste-video --source clip.mp4 --det-ckpt … --cls-ckpt …", "annotated.mp4, tracks.csv, inventory.txt"),
        ], [4.2, 6.3, 4.0], {"font": 9}),
        ("h2", "6.2 Classification and Routing of All 33 Classes"),
        ("p", "{F:c1}, {F:c2} and {F:c3} show one random held-out image of every one of the 33 classes. Each panel gives the true class, the most likely class and its confidence, the set "
              "of plausible families, the uncertainty U and the action taken. A title in orange marks a wrong most likely class."),
        ("fig", "c1", "classes_1", "Classes 1 to 11 of 33: Plastics, Newspaper and Magazines", 13.0, 19.0),
        ("fig", "c2", "classes_2", "Classes 12 to 22 of 33: Office Paper, Paper Cups, Cardboard, Glass and Metal", 13.0, 19.0),
        ("fig", "c3", "classes_3", "Classes 23 to 33 of 33: Organic, Styrofoam, Textile and the Three Hazardous Classes", 13.0, 19.0),
        ("p", "Of the 33 images, 29 have the correct most likely class. In three of the other four, the mix-up is between two classes of the same family (soda bottle read as water bottle, "
              "and boxes read as packaging and the reverse), so the item still goes automatically to the correct bin. In the last, an aluminium soda can read as plastic straws, the set of "
              "families is empty and the uncertainty is high, so the item goes to a person instead of to a wrong bin. {F:f1} shows the F1 score of each class on the whole test set."),
        ("fig", "f1", "per_class_f1", "F1 Score of Each of the 33 Classes on the 4,232 Test Images, Coloured by Material Family", 14.4),
        ("h2", "6.3 The Hazardous Family"),
        ("p", "{F:haz} shows batteries, electronic waste and medical waste. In the first two images of each class the system is sure and sends the item to the hazardous bin. In the others "
              "the most likely class is a neighbouring hazardous class (electronic waste read as medical waste) or even a non-hazardous class (a power strip read as a glass cosmetic "
              "container). Because the family set still contains the hazardous family, the item stays out of every recycling bin: it goes to the hazardous bin when the system is sure "
              "and to priority review (orange label) when it is not."),
        ("fig", "haz", "hazards", "The Hazardous Family: Confident Hazards and Hazards Whose Most Likely Class Differed From the True Class", 14.0, 19.0),
        ("h2", "6.4 Routing Decisions With Family Sets"),
        ("p", "{F:route} shows eight held-out images with their decisions. Two shoes and a food-waste item are confident, have one family in the set and go straight to their bins. A USB "
              "cable is read as electronic waste with low uncertainty and goes to the hazardous bin. A power strip (true class electronic waste, but read first as a glass cosmetic "
              "container) and a face mask (true class medical waste, but read first as an aluminium food can) still reach the hazardous bin, because the family set contains only the "
              "hazardous family; this is the case that a single most likely class would miss. A garment with a mixed set of two families, and a pair of shoes whose set is empty and "
              "whose uncertainty is high, go to a person."),
        ("fig", "route", "routing_demo", "Routing Decisions on Eight Held-Out Images: True Class, Most Likely Class, Family Set, Uncertainty and Action", 14.2),
        ("h2", "6.5 Explanations"),
        ("p", "{F:expl} shows the three kinds of explanation for one battery image: a Grad-CAM heat map, LIME superpixel outlines and a SHAP breakdown of a simulated contamination score. "
              "The heat map falls on the two battery bodies, which is the part of the picture that should decide the class, and the two image methods agree. In the SHAP bars the "
              "base value is the average score (0.666) and each simulated sensor reading adds or subtracts from it (moisture −0.693, gas −0.333), so the explanation adds up exactly to the "
              "reported score. {F:gc} shows the heat maps for 16 classes from all nine families; in most pairs the red areas fall on the object itself, which is evidence that the "
              "network uses the item's appearance and not the background."),
        ("fig", "expl", "explain_panel", "Explanation Views of One Battery Image: Heat Map, Superpixel Outlines and SHAP Bars (Simulated Sensors)", 14.4),
        ("fig", "gc", "gradcam", "Grad-CAM on 16 Classes From All Nine Families: Model Input and Heat Map for Each", 14.4),
        ("h2", "6.6 Detection and Item Outline"),
        ("p", "{F:det} shows the 18-class detector on held-out benchmark photographs, including pictures with several items. Each box carries the object name that feeds the cross-check of "
              "Section 4.1.2. {F:sam} shows the Segment Anything Model outline inside each box and the background-removed cut-out that can be passed to the classifier; the outline fills "
              "between 42% and 83% of the box in these six examples."),
        ("fig", "det", "detector", "The 18-Class Object Detector on 12 Held-Out Benchmark Photographs", 14.2),
        ("fig", "sam", "sam", "Item Outline of the Segment Anything Model (Top) and Background-Removed Cut-Out (Bottom)", 14.4),
        ("h2", "6.7 Video Litter Survey"),
        ("p", "The video survey was run on a 240-frame test video made from 12 real litter photographs shown for 20 frames each. The detector made 160 detections over the 240 frames; "
              "the tracker merged them into 16 inventory entries, each with one object type, one material and one routing decision. The clip has no ground-truth count of items, so "
              "the figure shows de-duplication and not counting accuracy. {F:vf} shows six annotated frames, {F:v12} one frame from each of the 12 scenes, and {F:inv} the inventory "
              "report. The written report that the pipeline produces is reproduced below."),
        ("fig", "vf", "video_frames", "Annotated Frames of the Video Survey: Each Box Carries the Track Number, Object Name and Confidence", 14.0),
        ("fig", "v12", "survey12", "One Annotated Frame From Each of the 12 Scenes of the Test Video", 14.4),
        ("fig", "inv", "survey_inv", "Inventory of the Video Survey by Object Type, Material and Routing Decision, and Frames Followed per Entry", 14.4),
        ("codefile", "reports/demo/video/inventory.txt", 40),
        ("p", "All 16 entries went to manual review in this run. This is the designed behaviour on field footage that looks different from the training photographs: the uncertainty "
              "is high and the gate sends items to a person, which is why the report counts 12 hazard-suspected entries that were not acted on automatically."),
        ("h2", "6.8 Class-Tour Video"),
        ("p", "A short video steps through all 33 classes with the prediction, family set, uncertainty and a coloured routing banner for each image. {F:tour} shows every third class; the "
              "full video is kept as a companion file (reports/demo/gallery/class_tour.mp4)."),
        ("fig", "tour", "tour_strip", "Frames of the Class-Tour Video: Every Third Class of the 33-Class Tour", 14.2),
        ("h2", "6.9 Contamination Index on Simulated Sensors"),
        ("p", "{F:ocil} shows how the Organic Contamination Index rises with the amount of residue left on an item, for the combined model and for each single-channel model, using the "
              "simulated sensor generator (two item categories, five residue levels from 0 g to 20 g, held-out draw). Residues of 5 g and more are counted as contaminated. "
              "The median index of the combined model rises from 0.36 for a clean item to 0.92 for 20 g of residue, and both single-channel models show the same steady rise (moisture only 0.46 to 0.85, "
              "gas only 0.47 to 0.93), which is why the system can switch to a single-channel model when a sensor stops working and still give a meaningful score. "
              "These are simulated sensor values; the physical sensors are the next build."),
        ("fig", "ocil", "oci_levels", "Organic Contamination Index Against Residue for Three Models (Simulated Sensors)", 14.4),
        ("h2", "6.10 Generated Training Images"),
        ("p", "{F:gan} shows images produced by the generative adversarial network for the textile class. The network learns the general shape and colour layout of clothing and can add "
              "extra training images for classes with few real photographs. The images are small and soft, so they are intended only as additional training examples next to real "
              "photographs, and their quality has not yet been measured with a standard score."),
        ("fig", "gan", "gan", "Images Generated by the Deep Convolutional Generative Adversarial Network for the Textile Class", 6.5),
        ("h2", "6.11 Reproducing the Demonstrations"),
        ("p", "Every number and figure in Chapters 6 and 7 can be regenerated. The commands are given in the repository's readme and in Appendix A, and the random seeds are fixed "
              "(42 for data and training, 0 for analysis resampling). Analysis inference runs in 32-bit floating point, because 16-bit precision flipped one of the 4,232 predictions."),
    ]
    return B
