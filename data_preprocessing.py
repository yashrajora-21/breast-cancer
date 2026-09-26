"""
data_preprocessing.py
----------------------
Handles data loading, train/test splitting, feature scaling, and class-imbalance
correction for the breast cancer detection project.

Dataset: Wisconsin Diagnostic Breast Cancer (WDBC) dataset, bundled with
scikit-learn (569 samples, 30 numeric diagnostic features, 2 classes:
malignant / benign). This is the same type of tabular diagnostic data
(cell nuclei measurements from a fine needle aspirate) referenced by the
project brief.

Because the raw WDBC dataset is only mildly imbalanced (~63% benign /
37% malignant), and the project brief specifically calls out "data
imbalance" as a challenge to be addressed, this module can optionally
down-sample the minority class to create a more strongly imbalanced,
realistic scenario, and then correct it on the TRAINING split only
using a custom SMOTE (Synthetic Minority Oversampling Technique)
implementation (no internet access is available in this environment to
install the `imbalanced-learn` package, so SMOTE is implemented here
from scratch using k-nearest-neighbour interpolation, which is exactly
how the original algorithm works).
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


RANDOM_STATE = 42


def load_raw_data():
    """Load the WDBC dataset as (X, y, feature_names, target_names)."""
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names, data.target_names


def induce_imbalance(X, y, minority_class=0, keep_fraction=0.35, random_state=RANDOM_STATE):
    """
    Artificially reduce the minority class ('malignant' = 0 in sklearn's
    encoding) so the dataset exhibits a realistic, pronounced class
    imbalance -- mirroring the imbalance problem described in the project
    brief. This is done BEFORE the train/test split so both the training
    and evaluation data reflect a real-world imbalanced scenario.
    """
    rng = np.random.RandomState(random_state)
    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y != minority_class)[0]

    n_keep = int(len(minority_idx) * keep_fraction)
    keep_minority = rng.choice(minority_idx, size=n_keep, replace=False)

    keep_idx = np.concatenate([majority_idx, keep_minority])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]


def smote_oversample(X, y, minority_class=0, k_neighbors=5, random_state=RANDOM_STATE):
    """
    A from-scratch implementation of SMOTE.

    For every minority-class sample, find its k nearest minority-class
    neighbours and generate synthetic samples along the line segments
    joining the sample to a randomly chosen neighbour, until the minority
    class matches the majority class count.
    """
    rng = np.random.RandomState(random_state)

    minority_mask = (y == minority_class)
    X_min = X[minority_mask]
    n_minority = len(X_min)
    n_majority = np.sum(~minority_mask)
    n_to_generate = n_majority - n_minority

    if n_to_generate <= 0:
        return X, y  # already balanced or minority is larger

    k = min(k_neighbors, n_minority - 1)
    if k < 1:
        # too few minority samples to interpolate; fall back to duplication
        synthetic = X_min[rng.randint(0, n_minority, size=n_to_generate)]
    else:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        _, neighbor_idx = nn.kneighbors(X_min)

        synthetic = np.zeros((n_to_generate, X.shape[1]))
        for i in range(n_to_generate):
            sample_idx = rng.randint(0, n_minority)
            neighbor_choice = neighbor_idx[sample_idx, rng.randint(1, k + 1)]
            gap = rng.uniform(0, 1)
            synthetic[i] = X_min[sample_idx] + gap * (X_min[neighbor_choice] - X_min[sample_idx])

    X_balanced = np.vstack([X, synthetic])
    y_balanced = np.concatenate([y, np.full(n_to_generate, minority_class)])

    # shuffle so synthetic samples aren't all at the end
    shuffle_idx = rng.permutation(len(y_balanced))
    return X_balanced[shuffle_idx], y_balanced[shuffle_idx]


def prepare_data(test_size=0.25, apply_synthetic_imbalance=True, random_state=RANDOM_STATE):
    """
    Full preprocessing pipeline:
      1. Load raw WDBC data
      2. (optional) induce a stronger, realistic class imbalance
      3. train/test split (stratified, so the test set keeps the same
         imbalance ratio as reality -- we never balance the test set)
      4. scale features (fit scaler on train only, to avoid leakage)
      5. SMOTE-balance the TRAINING split only

    Returns a dict with all data splits + metadata needed downstream.
    """
    X, y, feature_names, target_names = load_raw_data()

    if apply_synthetic_imbalance:
        X, y = induce_imbalance(X, y, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_bal, y_train_bal = smote_oversample(X_train_scaled, y_train, random_state=random_state)

    return {
        "X_train_raw": X_train_scaled,
        "y_train_raw": y_train,
        "X_train_balanced": X_train_bal,
        "y_train_balanced": y_train_bal,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_names": feature_names,
        "target_names": target_names,
        "scaler": scaler,
    }


if __name__ == "__main__":
    d = prepare_data()
    print("Train (before balancing):", np.bincount(d["y_train_raw"]))
    print("Train (after SMOTE):     ", np.bincount(d["y_train_balanced"]))
    print("Test (untouched, realistic ratio):", np.bincount(d["y_test"]))
