# 🧠 Preictal Seizure Detection — Siena Scalp EEG

A complete end-to-end EEG signal processing and machine learning pipeline for preictal seizure detection, applied to the [Siena Scalp EEG Dataset](https://physionet.org/content/siena-scalp-eeg/1.0.0/) (PhysioNet).

---

## 📌 Project Overview

Epileptic seizure prediction from scalp EEG is a clinically significant problem — early warning of seizure onset could enable timely intervention for the ~30% of epilepsy patients who are drug-resistant. This project investigates whether the **preictal state** (the 30 seconds immediately before seizure onset) can be distinguished from the **interictal baseline** using frequency-domain and complexity-based EEG features.

The pipeline covers every stage from raw `.edf` signal loading through to validated machine learning classification, with a specific focus on **methodological rigour**: Leave-One-Subject-Out cross-validation, SMOTE for class imbalance, and an explicit data scarcity analysis.

---

## 📊 Key Results

| Evaluation | Best Model | AUC |
|---|---|---|
| Subject-Independent (LOSO-CV) | Random Forest | 0.61 ± 0.24 |
| Subject-Dependent (within-patient) | SVM (RBF) | 0.70 ± 0.29 |

**Central finding:** Cross-patient generalisation is severely limited by inter-subject variability. Subject-dependent performance is further constrained by data scarcity — the Siena dataset has a median of only 5 labelled segments per patient, and a Pearson correlation of *r* = 0.61 (*p* = 0.063) was observed between segment availability and best achievable AUC.

---

## 🗂️ Dataset

**Siena Scalp EEG Dataset** — publicly available on PhysioNet  
🔗 https://physionet.org/content/siena-scalp-eeg/1.0.0/

- 14 epileptic patients, continuous scalp EEG recordings
- European Data Format (.edf), sampled at 512 Hz
- Standard 10–20 electrode placement (29–32 channels per patient)
- Ground-truth seizure onset/offset annotations provided
- 12 patients yielded valid segments after preprocessing → **66 labelled segments** (33 preictal, 33 interictal)

You will need a `seizure_times.csv` file with the following columns:

```
patient_id, file_name, reg_start, seiz_start, seizure_index
```

Times in `HH:MM:SS` format.

---

## ⚙️ Pipeline

```
1_visualisation.py              → Raw EEG visualisation
2_pre-processing.py             → Notch filter, bandpass, average reference
3_feature-extraction-bandpower  → Spectral band power (delta/theta/alpha/beta)
3_feature-extraction-entropy    → Permutation entropy per channel
4_analysis-bandpower_bargraph   → Per-channel band power bar plots
4_analysis-bandpower_boxplot    → Band power boxplots + t-tests
4_analysis-entropy_bargraph     → Per-channel entropy bar plots
4_analysis-entropy_ttest-boxplot→ Entropy paired t-tests + boxplots
5_thresholds-entropy.py         → Rule-based entropy threshold calculation
6_detection.py                  → Rule-based preictal flagging
7_ml_classification.py          → LOSO-CV: RF, SVM, XGBoost
8_subject_dependent.py          → Subject-dependent baseline + comparison plot
9_data_sufficiency.py           → Data scarcity analysis + visualisation
```

---

## 🖼️ Results Figures

### Raw EEG: Preictal vs Interictal
| Preictal (PN09) | Interictal (PN09) |
|---|---|
| ![Preictal](PN09_seizure19_preictal.png) | ![Interictal](PN09_seizure20_interictal.png) |

Higher amplitude and channel-correlated activity visible in the preictal window compared to the more independent interictal baseline.

---

### Extracted Band Power Features
![Band Power Table](Screenshot_2026-05-29_at_10_57_03_AM.png)

Per-channel spectral band power values (delta, theta, alpha, beta) for PN00 Seizure 2 (preictal). Values span several orders of magnitude across channels.

---

### LOSO-CV ROC Curves
![ROC Curves](roc_curves_loso.png)

All three classifiers perform near chance under LOSO-CV (AUC 0.53–0.61), reflecting high inter-subject variability in preictal EEG signatures.

---

### Subject-Dependent vs Subject-Independent AUC
![SD vs LOSO](sd_vs_loso_comparison.png)

SVM shows the largest benefit in within-patient mode (0.70 vs 0.54). The gap between subject-dependent and LOSO performance quantifies the inter-subject generalisation problem.

---

### Data Scarcity per Patient
![Segments per Patient](segments_per_patient.png)

Only 3 of 12 patients have ≥8 segments (green). The majority are data-scarce (red), with a median of 5 segments per patient.

---

### Data Scarcity vs Classification Performance
![Segments vs AUC](segments_vs_auc.png)

Pearson *r* = 0.61 (*p* = 0.063) between segment count and best AUC. Patients with ≤4 segments cluster at chance-level performance.

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/siena-eeg-preictal.git
cd siena-eeg-preictal

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install mne pandas numpy scipy matplotlib \
            antropy imbalanced-learn xgboost scikit-learn
```

---

## 🚀 Usage

Run scripts in order. All scripts are self-contained and use paths relative to their own location.

```bash
# 1. Visualise raw EEG
python 1_visualisation.py

# 2. Preprocess and save segment plots
python 2_pre-processing.py

# 3. Extract features
python "3_feature-extraction-bandpowervalues.py"
python "3_feature-extraction-entropy_values.py"

# 4. Statistical analysis and plots
python "4_analysis-bandpower_bargraph.py"
python "4_analysis-bandpower_boxplot.py"
python "4_analysis-entropy_bargraph.py"
python "4_analysis-entropy_ttest-boxplot.py"

# 5. Threshold calculation
python 5-thresholds-entropy.py

# 6. Rule-based detection
python 6-detection.py

# 7. ML classification (LOSO-CV)
python 7_ml_classification.py

# 8. Subject-dependent baseline
python 8_subject_dependent.py

# 9. Data scarcity analysis
python 9_data_sufficiency.py
```

Outputs are saved to:
- `analysis/` — preprocessed segment plots, band power CSVs, entropy CSVs
- `ml_results/` — classification summaries, per-fold results, all figures

---

## 📁 Project Structure

```
project/
├── seizure_times.csv
├── 1_visualisation.py
├── 2_pre-processing.py
├── 3_feature-extraction-bandpowervalues.py
├── 3_feature-extraction-entropy_values.py
├── 4_analysis-bandpower_bargraph.py
├── 4_analysis-bandpower_boxplot.py
├── 4_analysis-entropy_bargraph.py
├── 4_analysis-entropy_ttest-boxplot.py
├── 5-thresholds-entropy.py
├── 6-detection.py
├── 7_ml_classification.py
├── 8_subject_dependent.py
├── 9_data_sufficiency.py
├── analysis/
│   ├── PN00_seizure1_preictal_bandpower.csv
│   ├── PN00_seizure1_interictal_bandpower.csv
│   ├── PN00_seizure1_preictal_perm_entropy.csv
│   └── ...
└── ml_results/
    ├── classification_summary.csv
    ├── per_fold_results.csv
    ├── sd_summary.csv
    ├── sd_per_patient_results.csv
    ├── roc_curves_loso.png
    ├── sd_vs_loso_comparison.png
    ├── segments_per_patient.png
    └── segments_vs_auc.png
```

---

## 🔬 Methods Summary

| Step | Detail |
|---|---|
| **Preprocessing** | Notch filter (50 Hz), bandpass (0.5–40 Hz), average reference |
| **Preictal window** | 30s immediately before seizure onset |
| **Interictal window** | 30s starting 150s before seizure onset |
| **Band power** | Welch PSD, trapezoidal integration over δ/θ/α/β bands |
| **Permutation entropy** | Order *m*=3, delay τ=1, normalised |
| **Feature vector** | 95 dimensions (19ch × 4 bands + 19ch entropy) |
| **Class imbalance** | SMOTE (training folds only) |
| **CV strategy** | Leave-One-Subject-Out (subject-independent) |
| **Within-patient** | Stratified 80/20 split, 5 repeats |
| **Classifiers** | Random Forest, SVM (RBF), XGBoost |
| **Metrics** | AUC, Accuracy, Sensitivity, Specificity, F1 |

---

## 💡 Why LOSO-CV?

With only 12 patients, random k-fold cross-validation would allow train and test data from the same patient to appear in both splits — artificially inflating performance metrics. LOSO-CV ensures the model is always tested on a completely unseen patient, which is the only scientifically valid evaluation for this problem size and reflects real-world clinical deployment.

---

## 🔭 Future Work

- **Transfer learning** from larger datasets (CHB-MIT Scalp EEG) to address cross-patient variability
- **Patient-adaptive models** that update continuously as new seizure data is collected
- **Deep learning** (EEGNet, TCN) with data augmentation to overcome per-patient scarcity
- **Extended features** — phase-amplitude coupling, Hjorth parameters, graph connectivity

---

## 📄 Report

A full research paper write-up (LaTeX, two-column IEEE style) is included as `siena_eeg_report.tex`, suitable for Overleaf compilation.

---

## 📚 References

- Detti et al. (2020). *EEG synchronization analysis for seizure prediction.* Processes, 8(7), 846.
- Bandt & Pompe (2002). *Permutation entropy.* Physical Review Letters, 88(17).
- Mormann et al. (2007). *Seizure prediction: The long and winding road.* Brain, 130(2).
- Gramfort et al. (2013). *MNE-Python.* Frontiers in Neuroscience, 7, 267.
- Goldberger et al. (2000). *PhysioNet.* Circulation, 101(23).

---

## 👩‍💻 Author

**Caitlin Leonard**  
Biosignals & Neural Engineering Project, 2025–2026
