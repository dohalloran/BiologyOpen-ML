#!/usr/bin/env python3
"""
train_activity_models.py

Train several classifiers on combined RDKit descriptors and generate:
  1. ROC curves for all models on a single plot (SVG)
  2. Bar chart subplots for multiple performance metrics (SVG)
  3. Confusion matrices for all models in a single figure (SVG)

Input:
  - combined_training.csv
    (last column must be "Class", with 0 = inactive, 1 = active)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    classification_report,
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_score_vector(model, X):
    """
    Get a continuous score for ROC curves:
      - If predict_proba exists: use P(class=1)
      - Else if decision_function exists: use that
      - Else fall back to predicted labels (0/1)
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return model.predict(X)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Fit model, compute predictions and metrics, and print a summary.
    Returns a dictionary with metrics and confusion matrix.
    """
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = get_score_vector(model, X_test)

    # Confusion matrix (assuming binary labels 0 and 1)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)  # sensitivity (for class 1)
    # Specificity = recall of negative class
    spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ROC / AUC (if y_score is not constant)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    print("Confusion matrix (rows = true, cols = predicted, labels=[0,1]):")
    print(cm)
    print(f"Accuracy   : {acc:.3f}")
    print(f"Precision  : {prec:.3f}")
    print(f"Recall     : {rec:.3f} (sensitivity)")
    print(f"Specificity: {spec:.3f}")
    print(f"F1 score   : {f1:.3f}")
    print(f"AUC        : {roc_auc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    return {
        "model": model,
        "cm": cm,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
    }


def plot_roc_curves(results, out_path):
    """
    Single ROC figure with one curve per model.
    Saved as an SVG.
    """
    plt.figure(figsize=(7, 6))

    for name, res in results.items():
        plt.plot(
            res["fpr"],
            res["tpr"],
            lw=2,
            label=f"{name} (AUC = {res['auc']:.3f})",
        )

    # Chance diagonal
    plt.plot([0, 1], [0, 1], "k--", lw=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves for activity classification models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    print(f"Saved ROC figure to: {out_path}")


def plot_metric_bars(results, out_path):
    """
    A single figure with multiple bar chart subplots:
      - Accuracy
      - Precision
      - Recall (Sensitivity)
      - Specificity
      - F1 score
      - AUC

    Each subplot is a bar chart; all saved together as one SVG.
    """
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]
    titles = [
        "Accuracy",
        "Precision",
        "Recall (Sensitivity)",
        "Specificity",
        "F1 score",
        "AUC",
    ]

    model_names = list(results.keys())
    x = np.arange(len(model_names))

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(zip(metric_names, titles)):
        ax = axes[i]
        values = [results[m][metric] for m in model_names]

        bars = ax.bar(x, values)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)

        # Add data labels on top of bars
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # If any extra axes exist (in case of fewer metrics), remove them
    for j in range(len(metric_names), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Performance metrics for all models", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"Saved metric bar charts (subplots) to: {out_path}")


def plot_confusion_matrices(results, out_path):
    """
    Single figure with confusion matrices for all models as subplots.

    Each subplot is a 2x2 confusion matrix heatmap labeled:
      - True labels on y-axis
      - Predicted labels on x-axis
      - Class 0 = inactive, Class 1 = active

    The colorbar (legend) is placed on the right, in its own axis,
    so it doesn't overlap any of the matrices.
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    n_cols = 3  # arrange in up to 3 columns
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Handle case where there's only 1 row
    if n_rows == 1:
        axes = np.array(axes).reshape(1, -1)
    axes = axes.flatten()

    vmax = max(res["cm"].max() for res in results.values())

    im = None
    for idx, (name, res) in enumerate(results.items()):
        cm = res["cm"]
        ax = axes[idx]

        im = ax.imshow(
            cm,
            interpolation="nearest",
            cmap=plt.cm.Blues,
            vmin=0,
            vmax=vmax
        )
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Inactive (0)", "Active (1)"], rotation=45, ha="right")
        ax.set_yticklabels(["Inactive (0)", "Active (1)"])
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10,
                )

    # Remove unused axes if there are any
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # Layout: reserve some space on the right for the colorbar
    fig.suptitle("Confusion matrices for all models", fontsize=16, y=0.98)
    # Leave room on the right (0.9 instead of 1.0)
    fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])

    # Dedicated axis for the colorbar on the right
    # [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"Saved confusion matrix figure to: {out_path}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Input file: combined_training.csv in the same directory
    data_path = "combined_training.csv"

    # Output directory for SVG figures
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    dataset = pd.read_csv(data_path)

    # X = all descriptor columns, y = last column ("Class")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    print(f"Dataset shape: X = {X.shape}, y = {y.shape}")

    # Impute missing values (if any) with column means
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    X = imputer.fit_transform(X)

    # 80/20 stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    model_defs = {
        "Linear SVM": SVC(kernel="linear", probability=True, random_state=0),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
        "RBF SVM": SVC(kernel="rbf", probability=True, random_state=0),
        "Gaussian NB": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=10, criterion="entropy", random_state=0
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", random_state=0
        ),
    }

    # Train + evaluate all models
    results = {}
    for name, model in model_defs.items():
        results[name] = evaluate_model(name, model, X_train, X_test, y_train, y_test)

    # 1) ROC curves on one plot
    roc_path = os.path.join(out_dir, "activity_models_ROC_all.svg")
    plot_roc_curves(results, roc_path)

    # 2) Bar chart subplots (one figure)
    bars_path = os.path.join(out_dir, "activity_models_metrics_bars_subplots.svg")
    plot_metric_bars(results, bars_path)

    # 3) Confusion matrices for all models in one figure
    cm_path = os.path.join(out_dir, "activity_models_confusion_matrices_all.svg")
    plot_confusion_matrices(results, cm_path)

    print("\nDone.")
    print("SVG figures saved in:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
