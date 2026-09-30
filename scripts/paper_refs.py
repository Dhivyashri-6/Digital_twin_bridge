"""Reference list (30 entries, numbering as supplied by the authors), introduction paragraphs with the citations mapped
to that list, and the expanded related-work section. Venue names are set in italics (BPASTS/IEEE style)."""
from __future__ import annotations

# (authors+title before venue, venue in italics, text after venue)
REFERENCES = [
    ("R.P. Adams and D.J.C. MacKay, “Bayesian Online Changepoint Detection,” ", "arXiv preprint arXiv:0710.3742", ", 2007."),
    ("S.M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” ",
     "Advances in Neural Information Processing Systems", ", vol. 30, pp. 4765–4774, 2017."),
    ("T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in ",
     "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",
     ", 2016, pp. 785–794, doi: 10.1145/2939672.2939785."),
    ("E. Figueiredo, G. Park, C.R. Farrar, K. Worden, and J. Figueiras, “Machine Learning Algorithms for Damage Detection "
     "under Operational and Environmental Variability,” ", "Structural Health Monitoring",
     ", vol. 10, no. 6, pp. 559–572, 2011, doi: 10.1177/1475921710388971."),
    ("A. Santos, E. Figueiredo, M.F.M. Silva, C.S. Sales, and J.C.W.A. Costa, “Machine Learning Algorithms for Damage "
     "Detection: Kernel-Based Approaches,” ", "Journal of Sound and Vibration", ", vol. 363, pp. 584–599, 2016."),
    ("C.R. Farrar and K. Worden, “An Introduction to Structural Health Monitoring,” ",
     "Philosophical Transactions of the Royal Society A", ", vol. 365, no. 1851, pp. 303–315, 2007, doi: 10.1098/rsta.2006.1928."),
    ("K. Worden and J.M. Dulieu-Barton, “An Overview of Intelligent Fault Detection in Systems and Structures,” ",
     "Structural Health Monitoring", ", vol. 3, no. 1, pp. 85–98, 2004, doi: 10.1177/1475921704041866."),
    ("S.W. Doebling, C.R. Farrar, M.B. Prime, and D.W. Shevitz, “Damage Identification and Health Monitoring of Structural "
     "and Mechanical Systems from Changes in Their Vibration Characteristics: A Literature Review,” ",
     "Los Alamos National Laboratory Report", " LA-13070-MS, 1996."),
    ("E.P. Carden and P. Fanning, “Vibration Based Condition Monitoring: A Review,” ", "Structural Health Monitoring",
     ", vol. 3, no. 4, pp. 355–377, 2004, doi: 10.1177/1475921704047500."),
    ("O. Avci, O. Abdeljaber, S. Kiranyaz, M. Hussein, M. Gabbouj, and D.J. Inman, “A Review of Vibration-Based Damage "
     "Detection in Civil Structures: From Traditional Methods to Machine Learning and Deep Learning Applications,” ",
     "Mechanical Systems and Signal Processing", ", vol. 147, p. 107077, 2021, doi: 10.1016/j.ymssp.2020.107077."),
    ("R. Hou and Y. Xia, “Review on the New Development of Vibration-Based Damage Identification for Civil Engineering "
     "Structures: 2010–2019,” ", "Journal of Sound and Vibration", ", vol. 491, p. 115741, 2021, doi: 10.1016/j.jsv.2020.115741."),
    ("A. Malekloo, E. Ozer, M. AlHamaydeh, and M. Girolami, “Machine Learning and Structural Health Monitoring Overview with "
     "Emerging Technology and High-Dimensional Data Source Highlights,” ", "Structural Health Monitoring",
     ", vol. 21, no. 4, pp. 1906–1955, 2022, doi: 10.1177/14759217211036880."),
    ("L. Sun, Z. Shang, Y. Xia, S. Bhowmick, and S. Nagarajaiah, “Review of Bridge Structural Health Monitoring Aided by Big "
     "Data and Artificial Intelligence: From Condition Assessment to Damage Detection,” ", "Journal of Structural Engineering",
     ", vol. 146, no. 5, 2020, doi: 10.1061/(ASCE)ST.1943-541X.0002535."),
    ("J. Maeck and G. De Roeck, “Description of Z24 Benchmark,” ", "Mechanical Systems and Signal Processing",
     ", vol. 17, no. 1, pp. 127–131, 2003, doi: 10.1006/mssp.2002.1548."),
    ("B. Peeters and G. De Roeck, “One-Year Monitoring of the Z24-Bridge: Environmental Effects versus Damage Events,” ",
     "Earthquake Engineering & Structural Dynamics",
     ", vol. 30, no. 2, pp. 149–171, 2001, doi: 10.1002/1096-9845(200102)30:2<149::AID-EQE1>3.0.CO;2-Z."),
    ("J. Maeck, B. Peeters, and G. De Roeck, “Damage Identification on the Z24 Bridge Using Vibration Monitoring,” ",
     "Smart Materials and Structures", ", vol. 10, no. 3, pp. 512–517, 2001, doi: 10.1088/0964-1726/10/3/313."),
    ("B. Peeters and G. De Roeck, “Reference-Based Stochastic Subspace Identification for Output-Only Modal Analysis,” ",
     "Mechanical Systems and Signal Processing", ", vol. 13, no. 6, pp. 855–878, 1999, doi: 10.1006/mssp.1999.1249."),
    ("M.M.R. Taha, A. Noureldin, J.L. Lucero, and T.J. Baca, “Wavelet Transform for Structural Health Monitoring: A "
     "Compendium of Uses and Features,” ", "Structural Health Monitoring",
     ", vol. 5, no. 3, pp. 267–295, 2006, doi: 10.1177/1475921706067741."),
    ("H. Sohn, C.R. Farrar, F.M. Hemez, J.J. Czarnecki, D.D. Shunk, D.W. Stinemates, and B.R. Nadler, “A Review of "
     "Structural Health Monitoring Literature: 1996–2001,” ", "Los Alamos National Laboratory Report", " LA-13976-MS, 2004."),
    ("C.R. Farrar, H. Sohn, G. Park, et al., “A Statistical Pattern Recognition Paradigm for Structural Health Monitoring,” in ",
     "Proceedings of the 9th ASCE Specialty Conference on Probabilistic Mechanics and Structural Reliability", ", 2004."),
    ("Q. Lu, X. Xie, A.K. Parlikad, and J.M. Schooling, “Digital Twin-Enabled Anomaly Detection for Built Asset Monitoring "
     "in Operation and Maintenance,” ", "Automation in Construction", ", vol. 118, p. 103277, 2020."),
    ("M. Torzoni, M. Tezzele, S. Mariani, A. Manzoni, and K.E. Willcox, “A Digital Twin Framework for Civil Engineering "
     "Structures,” ", "Computer Methods in Applied Mechanics and Engineering", ", vol. 418, p. 116584, 2024."),
    ("Z. Sun, S. Jayasinghe, A. Sidiq, F. Shahrivar, M. Mahmoodian, and S. Setunge, “Approach Towards the Development of "
     "Digital Twin for Structural Health Monitoring of Civil Infrastructure: A Comprehensive Review,” ", "Sensors",
     ", vol. 25, no. 1, p. 59, 2025, doi: 10.3390/s25010059."),
    ("D. Jones, C. Snider, A. Nassehi, J. Yon, and B. Hicks, “Characterising the Digital Twin: A Systematic Literature "
     "Review,” ", "CIRP Journal of Manufacturing Science and Technology", ", vol. 29, pp. 36–52, 2020, doi: 10.1016/j.cirpj.2020.02.002."),
    ("W. Kritzinger, M. Karner, G. Traar, J. Henjes, and W. Sihn, “Digital Twin in Manufacturing: A Categorical Literature "
     "Review and Classification,” ", "IFAC-PapersOnLine", ", vol. 51, no. 11, pp. 1016–1022, 2018, doi: 10.1016/j.ifacol.2018.08.474."),
    ("A. Jiménez Rios, V. Plevris, and M. Nogal, “Bridge Management Through Digital Twin-Based Anomaly Detection Systems: "
     "A Systematic Review,” ", "Frontiers in Built Environment", ", vol. 9, p. 1176621, 2023, doi: 10.3389/fbuil.2023.1176621."),
    ("I.A. Nhamage, C.S. Horas, N.-S. Dang, J.A. Campos e Matos, and J. Poças Martins, “Strategies for Maximising the Value "
     "of Digital Twins for Bridge Management and Structural Monitoring: A Systematic Review,” ",
     "Archives of Computational Methods in Engineering", ", vol. 32, pp. 4555–4586, 2025, doi: 10.1007/s11831-025-10280-1."),
    ("M.M. Yousuf, T.J. Saravanan, and S.R. Dash, “Advancing Bridge Health Assessment and Management Through Digital Twins: "
     "A Comprehensive Review,” ", "Structural Health Monitoring", ", 2026, doi: 10.1177/14759217251403381."),
    ("R.S. Chaudhari, S. Nama, M.M. Narkhede, V.P. Bhangale, R.B. Gaikwad, and M.M. More, “Next Generation Vibration-Based "
     "Structural Health Monitoring for Bridges: A Systematic Review of AI, IoT, and Digital Twin Integrations,” ",
     "Journal of Building Pathology and Rehabilitation", ", vol. 11, p. 136, 2026, doi: 10.1007/s41024-026-00810-0."),
    ("M. Franciosi, M. Kasser, and M. Viviani, “Digital Twins in Bridge Engineering for Streamlined Maintenance and Enhanced "
     "Sustainability,” ", "Automation in Construction", ", vol. 168, p. 105834, 2024, doi: 10.1016/j.autcon.2024.105834."),
]

# Introduction of "Final Paper 1" – wording unchanged, citation numbers mapped to the reference list above.
INTRO = {
    "Road bridges are among the most critical assets":
        "Road bridges are among the most critical assets of transport networks, and a growing share of them is operating "
        "beyond its original design life. Periodic visual inspection remains the main source of condition information, but "
        "it is labour-intensive, partly subjective and blind to changes that occur between visits. Structural health "
        "monitoring (SHM) addresses this gap by recording the dynamic response of a structure continuously and converting it "
        "into quantitative evidence about its state [6], [7].",
    "The premise of vibration-based SHM":
        "The premise of vibration-based SHM is that changes in stiffness, mass or boundary conditions alter the measured "
        "dynamic response, so that damage can be inferred from changes in vibration characteristics [8], [9]. Over the last "
        "decade, machine-learning and deep-learning methods have been applied extensively to this problem [10]–[13]. In "
        "practice, however, the analytical stages that an engineer needs – detecting when the behaviour changed, estimating "
        "what condition a measurement resembles and understanding why a model reached its output – are usually developed "
        "and evaluated separately.",
    "Digital twins offer a way to bring these stages together":
        "Digital twins offer a way to bring these stages together. A digital twin couples a physical asset with a virtual "
        "representation that is updated from measured data and supports decision making throughout operation and "
        "maintenance [22], [24], [25]. Recent reviews and frameworks for civil and bridge infrastructure [23], [26]–[30] "
        "emphasise the integration of sensing, analytics and ",
    "visualisation, but also note that many implementations":
        "visualisation, but also note that many implementations stop at data management. Lu et al. [21] showed that Bayesian "
        "change point detection can serve as the anomaly-detection engine of a digital twin for built assets, using pump "
        "data from a building. Their framework did not include a supervised classification stage or a model-explanation "
        "layer, and it was not applied to bridge vibration data.",
    "This paper describes a digital-twin monitoring prototype":
        "This paper describes a digital-twin monitoring prototype for bridge SHM that chains BOCPD, XGBoost, SHAP and a "
        "weighted risk score, and reports the exact results obtained when the implemented pipeline is run on the Z24 bridge "
        "benchmark [14], [15]. All numerical values in Section 5 are taken directly from the pipeline outputs or recomputed "
        "from the saved model and data. Section 2 reviews related work, Section 3 describes the benchmark and the data used, "
        "Section 4 presents the method, Section 5 reports the experimental results, Section 6 discusses the results and "
        "Section 7 concludes the paper.",
}

# Expanded related work (Section 2); every reference of the list is cited.
RELATED = {
    "2.1": [
        "Farrar and Worden [6] framed SHM as a statistical pattern-recognition problem comprising operational evaluation, "
        "data acquisition, feature extraction and statistical model development, and Farrar et al. [20] set out this "
        "paradigm in detail for vibration-based monitoring. Worden and Dulieu-Barton [7] reviewed intelligent fault "
        "detection and highlighted feature selection and data fusion as central issues. The early literature reviews of "
        "Doebling et al. [8] and Sohn et al. [19], together with the survey of Carden and Fanning [9], catalogued "
        "damage-identification methods based on modal parameters, frequency response functions and time-domain signal "
        "statistics, while Taha et al. [18] compiled the uses of the wavelet transform for denoising, feature extraction "
        "and damage detection.",
        "More recent surveys document the shift towards data-driven approaches. Avci et al. [10] traced the move from "
        "traditional methods to machine-learning and deep-learning techniques in civil structures, Hou and Xia [11] "
        "reviewed vibration-based damage identification developed between 2010 and 2019, Malekloo et al. [12] discussed "
        "machine learning with emerging sensing technologies and high-dimensional data, and Sun et al. [13] reviewed how "
        "big data and artificial intelligence support bridge monitoring from condition assessment to damage detection. "
        "Operational and environmental variability is the main obstacle for such methods: Figueiredo et al. [4] compared "
        "machine-learning algorithms that normalise damage-sensitive features for changing temperature and loading, and "
        "Santos et al. [5] extended the comparison to kernel-based algorithms that model nonlinear relationships between "
        "the features and the environment. A recurring observation in these studies is that environmental effects can "
        "mask or mimic damage-induced changes, so that reliable validation requires data from independent measurement "
        "campaigns.",
    ],
    "2.2": [
        "Kritzinger et al. [25] distinguished digital models, digital shadows and digital twins by the degree of automatic "
        "data exchange between the physical object and its virtual counterpart, and Jones et al. [24] characterised the "
        "digital twin through a systematic literature review of its constituent parts and open research questions. For "
        "civil structures, Torzoni et al. [22] proposed a digital-twin framework that combines physics-based reduced-order "
        "models with deep learning for structural health monitoring and decision making, and Sun et al. [23] reviewed the "
        "development of digital twins for SHM of civil infrastructure, covering sensing, data acquisition, modelling and "
        "health assessment.",
        "Bridge-specific studies have grown quickly in recent years. Franciosi et al. [30] examined digital twins for "
        "streamlined bridge maintenance and sustainability, Jiménez Rios et al. [26] reviewed digital-twin-based anomaly "
        "detection systems for bridge management, and Nhamage et al. [27] analysed strategies for maximising the value of "
        "digital twins in bridge management and structural monitoring. Yousuf et al. [28] surveyed digital twins for "
        "bridge health assessment and management, and Chaudhari et al. [29] reviewed how artificial intelligence, the "
        "Internet of Things and digital twins are being combined in vibration-based bridge monitoring. These works "
        "motivate pipelines in which sensing, analysis and visualisation share the same evidence, which is the design "
        "principle of the prototype described here.",
    ],
    "2.3": [
        "BOCPD, introduced by Adams and MacKay [1], maintains a posterior distribution over the run length – the number of "
        "observations since the last change – and updates it recursively as data arrive. Lu et al. [21] embedded Bayesian "
        "change point detection in a digital-twin-enabled anomaly-detection process for asset monitoring during operation "
        "and maintenance. XGBoost [3] is a regularised gradient-boosted tree ensemble that performs well on tabular feature "
        "data, and SHAP [2] provides additive, game-theoretic feature attributions whose exact tree variant makes it a "
        "natural companion for XGBoost.",
        "For bridges, the Z24 benchmark [14] and its one-year monitoring campaign [15] remain standard references, notably "
        "because they show how temperature-driven changes of the natural frequencies can be as large as damage-induced "
        "changes. Peeters and De Roeck [17] developed reference-based stochastic subspace identification, which estimates "
        "modal parameters from output-only measurements using a subset of reference sensors, and Maeck et al. [16] used "
        "vibration monitoring of the progressive damage tests to identify the induced damage. Taken together, the reviewed "
        "works treat change detection, condition classification and explanation largely as separate stages; the present "
        "study combines them in one digital-twin workflow and evaluates each stage on the complete Z24 progressive damage "
        "tests.",
    ],
}
