"""
train_evaluate.py
------------------
Main entry point for the Intelligent Breast Cancer Detection and
Classification project.

Pipeline:
  1. Load and preprocess data (scaling + SMOTE-based imbalance handling)
  2. Train the hybrid stacking ensemble
  3. Train baseline (solo) models for comparison
  4. Evaluate every model on the held-out, realistically-imbalanced test
     set using accuracy, precision, recall, F1-score, and ROC-AUC
  5. Save a comparison table, confusion matrix, and ROC curves to
     ./outputs/

Run with:  python3 train_evaluate.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from data_preprocessing import prepare_data
from hybrid_model import build_hybrid_model, build_baseline_models

OUTPUT_DIR = "outputs"


def evaluate_model(name, model, X_test, y_test, positive_label=1):
    """Fit-independent evaluation: assumes model is already trained."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, positive_label]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=positive_label),
        "recall": recall_score(y_test, y_pred, pos_label=positive_label),
        "f1_score": f1_score(y_test, y_pred, pos_label=positive_label),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


def main():
    print("=" * 70)
    print("Intelligent Breast Cancer Detection and Classification")
    print("Hybrid AI (Stacking Ensemble) Pipeline")
    print("=" * 70)

    # ---- 1. Data ----
    data = prepare_data(apply_synthetic_imbalance=True)
    X_train, y_train = data["X_train_balanced"], data["y_train_balanced"]
    X_test, y_test = data["X_test"], data["y_test"]
    target_names = data["target_names"]

    print(f"\nTraining set (after SMOTE balancing): {np.bincount(y_train)}")
    print(f"Test set (realistic, imbalanced):      {np.bincount(y_test)}")

    # ---- 2. Models ----
    models = build_baseline_models()
    models["HYBRID (Stacking Ensemble)"] = build_hybrid_model()

    results = []
    predictions = {}

    for name, model in models.items():
        print(f"\nTraining: {name} ...")
        model.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        predictions[name] = (y_pred, y_proba)
        print(
            f"  Accuracy={metrics['accuracy']:.4f}  "
            f"Precision={metrics['precision']:.4f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"F1={metrics['f1_score']:.4f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    # ---- 3. Save comparison table ----
    with open(f"{OUTPUT_DIR}/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown table for the report
    header = "| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |\n"
    header += "|---|---|---|---|---|---|\n"
    rows = ""
    for r in results:
        rows += (
            f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1_score']:.4f} | {r['roc_auc']:.4f} |\n"
        )
    with open(f"{OUTPUT_DIR}/model_comparison.md", "w") as f:
        f.write(header + rows)

    # ---- 4. Detailed report for the hybrid model ----
    hybrid_name = "HYBRID (Stacking Ensemble)"
    y_pred_hybrid, y_proba_hybrid = predictions[hybrid_name]
    report_text = classification_report(y_test, y_pred_hybrid, target_names=target_names)
    with open(f"{OUTPUT_DIR}/hybrid_classification_report.txt", "w") as f:
        f.write(f"Classification Report - {hybrid_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report_text)

    # ---- 5. Confusion matrix plot (hybrid model) ----
    cm = confusion_matrix(y_test, y_pred_hybrid)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix\n{hybrid_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/confusion_matrix_hybrid.png", dpi=150)
    plt.close(fig)

    # ---- 6. ROC curves for all models ----
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name in models:
        _, y_proba = predictions[name]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        style = "-" if "HYBRID" in name else "--"
        lw = 3 if "HYBRID" in name else 1.5
        ax.plot(fpr, tpr, style, linewidth=lw, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Hybrid Model vs. Baseline Learners")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/roc_curves_comparison.png", dpi=150)
    plt.close(fig)

    # ---- 7. Bar chart comparing F1 scores ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    names = [r["model"] for r in results]
    f1s = [r["f1_score"] for r in results]
    colors = ["#c0392b" if "HYBRID" in n else "#7f8c8d" for n in names]
    ax.barh(names, f1s, color=colors)
    ax.set_xlabel("F1-score")
    ax.set_title("Model Comparison: F1-score (malignant class)")
    ax.set_xlim(0, 1.0)
    for i, v in enumerate(f1s):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/f1_comparison.png", dpi=150)
    plt.close(fig)

    print("\n" + "=" * 70)
    print("Done. Results saved to ./outputs/:")
    print("  - model_comparison.json / model_comparison.md")
    print("  - hybrid_classification_report.txt")
    print("  - confusion_matrix_hybrid.png")
    print("  - roc_curves_comparison.png")
    print("  - f1_comparison.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
