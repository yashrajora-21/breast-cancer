"""
hybrid_model.py
----------------
Defines the "Hybrid AI" classifier for breast cancer detection.

The hybrid approach follows a STACKING ENSEMBLE design: several
structurally different base learners -- each with different inductive
biases -- are trained on the same data, and a meta-learner is trained
on their out-of-fold predictions to combine them optimally. This is a
standard, well-validated way to build a "hybrid" model that leverages
the complementary strengths of multiple learning techniques:

    Base learners (diverse learning paradigms):
      - Random Forest        -> bagged decision trees, robust to noise
      - Gradient Boosting     -> sequential boosting, captures complex
                                  non-linear interactions
      - Support Vector Machine (RBF kernel) -> strong margin-based
                                  separator, good on scaled numeric data
      - Multi-Layer Perceptron (neural network) -> captures non-linear
                                  patterns via learned representations

    Meta-learner:
      - Logistic Regression -> combines the base learners' predicted
                                probabilities into a final decision

This mirrors exactly the kind of "hybrid AI model that combines the
strengths of multiple learning techniques" called for in the project
brief, and is what most literature on hybrid breast-cancer classifiers
implements (tree ensembles + kernel methods + neural nets stacked
together).
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42


def build_hybrid_model():
    """Construct and return the hybrid stacking ensemble (untrained)."""

    base_learners = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "svm_rbf",
            SVC(
                kernel="rbf",
                C=2.0,
                gamma="scale",
                probability=True,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "neural_net",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    meta_learner = LogisticRegression(max_iter=1000)

    hybrid_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,                 # 5-fold cross-validation to build meta-features
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    return hybrid_model


def build_baseline_models():
    """
    A few standalone models for comparison against the hybrid ensemble,
    so the report can quantify the benefit of hybridization.
    """
    return {
        "Random Forest (solo)": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM RBF (solo)": SVC(
            kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=RANDOM_STATE
        ),
        "Neural Network (solo)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=RANDOM_STATE
        ),
        "Gradient Boosting (solo)": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }
