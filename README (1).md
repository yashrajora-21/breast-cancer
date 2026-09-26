# Intelligent Breast Cancer Detection and Classification Using Hybrid AI Techniques

**Domain:** AI / ML

This is a working implementation matching the project proposal: a hybrid AI
system for breast cancer detection that addresses **data imbalance**,
**distribution variation**, and **inconsistent classification performance**
using multiple learning techniques combined into one ensemble.

## 1. Dataset

The Wisconsin Diagnostic Breast Cancer (WDBC) dataset (569 patients, 30
numeric features computed from digitized images of fine needle aspirate
(FNA) of breast masses — radius, texture, perimeter, area, smoothness,
concavity, symmetry, etc. — with a benign/malignant label). It ships with
scikit-learn, so the project runs with no external downloads. You can swap
in a different CSV by editing `data_preprocessing.py::load_raw_data()`.

## 2. Addressing data imbalance

The brief specifically flags data imbalance as a core challenge. The
pipeline:
1. Optionally exaggerates the natural class imbalance (`induce_imbalance`)
   to simulate a realistic, harder scenario.
2. Splits data into train/test **before** any balancing, so the test set
   keeps a realistic, untouched imbalance ratio (this is what a deployed
   system would actually see).
3. Balances **only the training set** using a from-scratch **SMOTE**
   (Synthetic Minority Oversampling Technique) implementation
   (`smote_oversample`), generating synthetic minority samples via
   k-nearest-neighbour interpolation rather than naive duplication.

## 3. The hybrid AI model

`hybrid_model.py` builds a **stacking ensemble** — the standard way to
combine multiple learning techniques into one hybrid model:

| Role | Technique | Why it's included |
|---|---|---|
| Base learner | Random Forest | Robust to noise/outliers, handles non-linear feature interactions |
| Base learner | Gradient Boosting | Sequentially corrects errors, strong on tabular data |
| Base learner | SVM (RBF kernel) | Strong margin-based separator on scaled numeric features |
| Base learner | Neural Network (MLP) | Learns non-linear representations |
| Meta-learner | Logistic Regression | Learns how to best combine the 4 base learners' predictions |

Each base learner has a different inductive bias, so the ensemble is more
robust and generalizes better than any single technique — directly
addressing the brief's goal of improving "accuracy, reliability,
robustness, and generalization."

## 4. Evaluation

`train_evaluate.py` trains the hybrid model **and** each base learner alone
(for comparison), then reports, on the held-out imbalanced test set:
accuracy, precision, recall, F1-score, and ROC-AUC — exactly the metrics
named in the proposal.

## 5. How to run

```bash
pip install scikit-learn numpy matplotlib
python3 train_evaluate.py
```

Outputs land in `outputs/`:
- `model_comparison.md` / `.json` — metrics table, hybrid vs. baselines
- `hybrid_classification_report.txt` — per-class precision/recall/F1
- `confusion_matrix_hybrid.png`
- `roc_curves_comparison.png` — hybrid model vs. each solo learner
- `f1_comparison.png` — bar chart of F1-scores

## 6. File structure

```
breast_cancer_hybrid_ai/
├── data_preprocessing.py   # loading, imbalance handling, scaling, SMOTE
├── hybrid_model.py         # stacking ensemble + baseline models
├── train_evaluate.py       # main script: train, evaluate, plot, save
├── README.md
└── outputs/                # generated metrics + plots (after running)
```

## 7. Extending this into your full report/dissertation

- **Swap the dataset:** replace `load_raw_data()` with `pd.read_csv(...)`
  for a hospital-specific dataset (mammography, biopsy, or genomic data).
- **Add feature selection / dimensionality reduction** (e.g. PCA, mutual
  information) before scaling, if your final report wants to discuss
  "feature engineering" as a separate subsection.
- **Hyperparameter tuning:** wrap the hybrid model in
  `GridSearchCV`/`RandomizedSearchCV` for a "Model Optimization" section.
- **Explainability:** add SHAP or permutation importance on the Random
  Forest/Gradient Boosting base learners for a "Clinical Interpretability"
  section, which reviewers of medical AI projects often expect.
