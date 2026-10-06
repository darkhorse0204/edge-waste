# report_refs.py - apa 7th edition reference list of the project report (sorted automatically)
"""Every entry was checked against the publisher or Crossref record, an arXiv page or Google Patents.
Markup: <i>italic</i> for journal, book and proceedings titles with volume.  Names that could not be verified
in full are given as first author et al. rather than guessed."""
import re
import unicodedata

REFS = [
    "Alatawi, M. N. (2025). Edge computing and federated learning for privacy-preserving IoT analytics. <i>EURASIP Journal on Wireless Communications and Networking, 2026</i>(1), Article 6. https://doi.org/10.1186/s13638-025-02545-x",
    "Alkılınç, A., Okay, F., Kök, İ., & Özdemir, S. (2025). Deep ensemble learning model for waste classification systems. <i>Sustainability, 18</i>(1), Article 24. https://doi.org/10.3390/su18010024",
    "Alnanih, R., Elrefaei, L., & Al-Ahwal, A. (2025). Advancing sustainability through an IoT-driven smart waste management system with software engineering integration. <i>Sustainability, 17</i>(21), Article 9803. https://doi.org/10.3390/su17219803",
    "Alsabt, R., Alkhaldi, W., Adenle, Y. A., & Alshuwaikhat, H. M. (2024). Optimizing waste management strategies through artificial intelligence and machine learning – An economic and environmental impact study. <i>Cleaner Waste Systems, 8</i>, Article 100158. https://doi.org/10.1016/j.clwas.2024.100158",
    "Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. <i>Foundations and Trends in Machine Learning, 16</i>(4), 494–591. https://arxiv.org/abs/2107.07511",
    "Arun, M. (2025). Investigation of a deep learning-based waste recovery framework for sustainability and a clean environment using IoT. <i>Sustainable Food Technology, 3</i>(2), 599–611. https://doi.org/10.1039/d4fb00340c",
    "Bircanoğlu, C., Atay, M., Beşer, F., Genç, Ö., & Kızrak, M. A. (2018). RecycleNet: Intelligent waste sorting using deep neural networks. <i>2018 Innovations in Intelligent Systems and Applications (INISTA)</i>. https://doi.org/10.1109/INISTA.2018.8466276",
    "Casao, S., Peña, F., Sabater, A., Castillón, R., Suárez, D., Montijano, E., & Murillo, A. C. (2024). <i>SpectralWaste dataset: Multimodal data for waste sorting automation</i> (arXiv:2403.18033). arXiv. https://doi.org/10.48550/arXiv.2403.18033",
    "Chahine, K., & Ghazal, B. (2017). Automatic sorting of solid wastes using sensor fusion. <i>International Journal of Engineering and Technology, 9</i>(6), 4408–4414. https://doi.org/10.21817/ijet/2017/v9i6/170906127",
    "Chu, Y., Huang, C., Xie, X., Tan, B., Kamal, S., & Xiong, X. (2018). Multilayer hybrid deep-learning method for waste classification and recycling. <i>Computational Intelligence and Neuroscience, 2018</i>, Article 5060857. https://doi.org/10.1155/2018/5060857",
    "Cortes, C., & Vapnik, V. (1995). Support-vector networks. <i>Machine Learning, 20</i>(3), 273–297. https://doi.org/10.1007/BF00994018",
    "Dipo, M. H., Farid, F. A., Mahmud, M. S. A., Momtaz, M., Rahman, S., Uddin, J., & Karim, H. A. (2025). Real-time waste detection and classification using a YOLOv12-based deep learning model. <i>Digital, 5</i>(2), Article 19. https://doi.org/10.3390/digital5020019",
    "Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. <i>International Conference on Learning Representations (ICLR 2021)</i>. https://arxiv.org/abs/2010.11929",
    "Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. <i>Proceedings of the 33rd International Conference on Machine Learning, PMLR 48</i>, 1050–1059.",
    "Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. <i>Advances in Neural Information Processing Systems, 30</i>. https://arxiv.org/abs/1705.08500",
    "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. <i>Proceedings of the 34th International Conference on Machine Learning, PMLR 70</i>, 1321–1330.",
    "He, W., Jiang, Z., Xiao, T., Xu, Z., & Li, Y. (2026). A survey on uncertainty quantification methods for deep learning. <i>ACM Computing Surveys, 58</i>(7), Article 179. https://doi.org/10.1145/3786319",
    "Hendrycks, D., & Gimpel, K. (2016). <i>Gaussian error linear units (GELUs)</i> (arXiv:1606.08415). arXiv. https://arxiv.org/abs/1606.08415",
    "Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. <i>International Conference on Learning Representations (ICLR 2017)</i>. https://arxiv.org/abs/1610.02136",
    "Hussain, M., O'Nils, M., Lundgren, J., & Mousavirad, S. J. (2024). A comprehensive review on deep learning-based data fusion. <i>IEEE Access, 12</i>, 180093–180124. https://doi.org/10.1109/ACCESS.2024.3508271",
    "Kaza, S., Yao, L. C., Bhada-Tata, P., & Van Woerden, F. (2018). <i>What a waste 2.0: A global snapshot of solid waste management to 2050</i>. World Bank. https://doi.org/10.1596/978-1-4648-1329-0",
    "Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment anything. <i>Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)</i>, 4015–4026. https://arxiv.org/abs/2304.02643",
    "Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). Fine-tuning can distort pretrained features and underperform out-of-distribution. <i>International Conference on Learning Representations (ICLR 2022)</i>. https://arxiv.org/abs/2202.10054",
    "Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. In <i>Computer Vision – ECCV 2014</i> (pp. 740–755). Springer. https://doi.org/10.1007/978-3-319-10602-1_48",
    "Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. <i>Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)</i>, 10012–10022. https://arxiv.org/abs/2103.14030",
    "Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. <i>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 11976–11986. https://arxiv.org/abs/2201.03545",
    "Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. <i>International Conference on Learning Representations (ICLR 2019)</i>. https://arxiv.org/abs/1711.05101",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. <i>Advances in Neural Information Processing Systems, 30</i>. https://arxiv.org/abs/1705.07874",
    "McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. <i>Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, PMLR 54</i>, 1273–1282.",
    "Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2021). Long-tail learning via logit adjustment. <i>International Conference on Learning Representations (ICLR 2021)</i>. https://arxiv.org/abs/2007.07314",
    "Nahiduzzaman, M., Ahamed, M. F., Naznine, M., Karim, M. J., Kibria, H. B., Ayari, M. A., Khandakar, A., Ashraf, A., Ahsan, M., & Haider, J. (2025). An automated waste classification system using deep learning techniques: Toward efficient waste recycling and environmental sustainability. <i>Knowledge-Based Systems, 310</i>, Article 113028. https://doi.org/10.1016/j.knosys.2025.113028",
    "Partosan, K. N. L., Villanueva, E. J. R., & Garcia, R. G. (2026). Campus-scale real-time waste classification on Raspberry Pi 5 using You Only Look Once version 8. <i>Engineering Proceedings, 134</i>(1), Article 28. https://doi.org/10.3390/engproc2026134028",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825–2830.",
    "Proença, P. F., & Simões, P. (2020). <i>TACO: Trash annotations in context for litter detection</i> (arXiv:2003.06975). arXiv. https://arxiv.org/abs/2003.06975",
    "Radchenko, G., & Fill, V. A. (2024). <i>Uncertainty estimation in multi-agent distributed learning for AI-enabled edge devices</i> (arXiv:2403.09141). arXiv. https://arxiv.org/abs/2403.09141",
    "Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised representation learning with deep convolutional generative adversarial networks. <i>International Conference on Learning Representations (ICLR 2016)</i>. https://arxiv.org/abs/1511.06434",
    "Raghavendra, K., et al. (2026). Multimodal sensor fusion for waste management using graph neural networks. <i>Journal Européen des Systèmes Automatisés, 59</i>(1), 253–264. https://doi.org/10.18280/jesa.590122",
    "Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. <i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 779–788. https://arxiv.org/abs/1506.02640",
    "Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?”: Explaining the predictions of any classifier. <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, 1135–1144. https://doi.org/10.1145/2939672.2939778",
    "Ruiz, V., Sánchez, Á., Vélez, J. F., & Raducanu, B. (2019). Automatic image-based waste classification. <i>Lecture Notes in Computer Science, 11487</i>, 422–431. https://doi.org/10.1007/978-3-030-19651-6_41",
    "Sadinle, M., Lei, J., & Wasserman, L. (2019). Least ambiguous set-valued classifiers with bounded error levels. <i>Journal of the American Statistical Association, 114</i>(525), 223–234. https://doi.org/10.1080/01621459.2017.1395341",
    "Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. <i>Proceedings of the IEEE International Conference on Computer Vision (ICCV)</i>, 618–626. https://arxiv.org/abs/1610.02391",
    "Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception architecture for computer vision. <i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 2818–2826. https://doi.org/10.1109/CVPR.2016.308",
    "van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. <i>Journal of Machine Learning Research, 9</i>, 2579–2605.",
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. <i>Advances in Neural Information Processing Systems, 30</i>, 5998–6008. https://arxiv.org/abs/1706.03762",
    "Verber, D., Grneva, T., & Dugonik, J. (2026). Image-based waste classification using a hybrid deep learning architecture with transfer learning and edge AI deployment. <i>Mathematics, 14</i>(7), Article 1176. https://doi.org/10.3390/math14071176",
    "Vovk, V. (2012). Conditional validity of inductive conformal predictors. <i>Proceedings of the Asian Conference on Machine Learning, PMLR 25</i>, 475–490.",
    "Vovk, V., Gammerman, A., & Shafer, G. (2005). <i>Algorithmic learning in a random world</i>. Springer. https://doi.org/10.1007/b106715",
    "White, G., Cabrera, C., Palade, A., Li, F., & Clarke, S. (2020). <i>WasteNet: Waste classification at the edge for smart bins</i> (arXiv:2006.05873). arXiv. https://arxiv.org/abs/2006.05873",
    "Yang, M., & Thung, G. (2016). <i>Classification of trash for recyclability status</i> (CS229 project report). Stanford University.",
    "Zhang, C., Han, D., Qiao, Y., Kim, J. U., Bae, S.-H., Lee, S., & Hong, C. S. (2023). <i>Faster Segment Anything: Towards lightweight SAM for mobile applications</i> (arXiv:2306.14289). arXiv. https://arxiv.org/abs/2306.14289",
    "Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. In <i>Computer Vision – ECCV 2022</i> (pp. 1–21). Springer. https://arxiv.org/abs/2110.06864",
    "ISO/IEC/IEEE. (2018). <i>Systems and software engineering – Life cycle processes – Requirements engineering</i> (ISO/IEC/IEEE Standard No. 29148:2018). International Organization for Standardization.",
    "Ministry of Environment, Forest and Climate Change. (2016a). <i>Solid Waste Management Rules, 2016</i>. Government of India.",
    "Ministry of Environment, Forest and Climate Change. (2016b). <i>Bio-Medical Waste Management Rules, 2016</i>. Government of India.",
    "Ministry of Environment, Forest and Climate Change. (2022a). <i>E-Waste (Management) Rules, 2022</i>. Government of India.",
    "Ministry of Environment, Forest and Climate Change. (2022b). <i>Battery Waste Management Rules, 2022</i>. Government of India.",
    "United Nations Institute for Training and Research. (2024). <i>Global e-waste monitor 2024: Electronic waste rising five times faster than documented e-waste recycling</i> [Press release]. https://unitar.org/about/news-stories/press/global-e-waste-monitor-2024-electronic-waste-rising-five-times-faster-documented-e-waste-recycling",
]

PATENTS = [
    "Battelle Energy Alliance, LLC, & Sortera Technologies, Inc. (2024). <i>Sorting of plastics</i> (U.S. Patent No. 11,969,764). U.S. Patent and Trademark Office. https://patents.google.com/patent/US11969764B2/en",
    "CleanRobotics Technologies, Inc. (2021). <i>Automatic sorting of waste</i> (U.S. Patent No. 11,104,512). U.S. Patent and Trademark Office. https://patents.google.com/patent/US11104512B2",
    "Commonwealth Scientific and Industrial Research Organisation. (2024). <i>Waste sorting method and apparatus</i> (International Publication No. WO 2024/207048 A1). World Intellectual Property Organization. https://patents.google.com/patent/WO2024207048A1/en",
    "Fidelity AG Inc. (2022). <i>Methods and electronic devices for automated waste management</i> (U.S. Patent No. 11,335,086). U.S. Patent and Trademark Office. https://patents.google.com/patent/US11335086B2/en",
    "Heil Co. (2024). <i>Refuse contamination analysis</i> (U.S. Patent No. 11,875,301). U.S. Patent and Trademark Office. https://patents.google.com/patent/US11875301B2/en",
    "Horowitz, M. B., Bailey, J. A., & McCoy, J. C., Jr. (2020). <i>Systems and methods for sorting recyclable items and other materials</i> (U.S. Patent No. 10,799,915). U.S. Patent and Trademark Office. https://patents.google.com/patent/US10799915B2/en",
    "IP Australia. (2021). <i>The waste segregation method using machine learning technique</i> (Australian Innovation Patent No. 2021101744). https://patents.google.com/patent/AU2021101744A4/en",
    "Prince Mohammad Bin Fahd University. (2024). <i>Waste management system</i> (U.S. Patent No. 11,961,054). U.S. Patent and Trademark Office.",
    "Tomra Sorting GmbH. (2025). <i>Neural network for bulk sorting</i> (European Patent No. 4,055,520 B1). European Patent Office. https://patents.google.com/patent/EP4055520B1",
]

WEBSITES = [
    "Google. (n.d.). <i>Google Colaboratory</i>. https://colab.research.google.com/",
    "ONNX. (n.d.). <i>Open Neural Network Exchange</i>. https://onnx.ai/",
    "Ultralytics. (n.d.). <i>Ultralytics YOLO documentation</i>. https://docs.ultralytics.com/",
    "Wightman, R. (2019). <i>PyTorch image models</i> [Computer software]. GitHub. https://github.com/huggingface/pytorch-image-models",
]

DATASETS = [
    "alistairking. (n.d.). <i>Recyclable and household waste classification</i> [Data set]. Kaggle. https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification",
    "minhle13. (n.d.). <i>TrashBox</i> [Data set]. Kaggle. https://www.kaggle.com/datasets/minhle13/trashbox",
    "mostafaabla. (n.d.). <i>Garbage classification (12 classes)</i> [Data set]. Kaggle. https://www.kaggle.com/datasets/mostafaabla/garbage-classification",
    "Proença, P. F., & Simões, P. (2020). <i>TACO: Trash annotations in context</i> [Data set]. http://tacodataset.org/",
]


def _key(s: str) -> str:
    t = re.sub(r"<[^>]+>", "", s).lower()
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))


def sorted_refs(lst):
    return sorted(lst, key=_key)
