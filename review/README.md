<!-- README.md - what is inside the review folder and in which order to open it -->

# review folder

| file or folder | what it is |
|---|---|
| `Technical_Dossier.docx` / `.pdf` | full technical dossier in simple english (what, how, why, where the code is, results, limits, 26 likely questions with answers) |
| `notebooks/06_colab_training_executed_annotated.ipynb` | **the real Colab training run** (outputs kept) with simple comments and "what / where / why" notes. open this first if the code notebook is asked |
| `notebooks/01_data_and_taxonomy.ipynb` | classes, split, imbalance, augmentation |
| `notebooks/02_model_code_and_training.ipynb` | model code, parameters, training settings, training curves |
| `notebooks/03_results_and_analysis.ipynb` | accuracy, confusion matrix, calibration, imbalance, pca / lda / svm baselines |
| `notebooks/04_safety_mechanisms.ipynb` | uncertainty, conformal routing, cross-check, contamination index, decision engine |
| `notebooks/05_live_demo_federated_video.ipynb` | full pipeline on real images, grad-cam, federated simulation, video inventory |
| `supporting/` | project report (docx, pdf), review slides (25), esp32 hardware guide |
| `media/` | two short demo videos (33-class tour, litter survey) |

notebooks 01-05 were executed on this laptop (outputs are saved inside them). they find the project folder by themselves; run them with the project folder as the base (jupyter lab from the project folder).
the original executed colab file is `notebooks/colab_classifier_training_executed.ipynb` (not changed).
