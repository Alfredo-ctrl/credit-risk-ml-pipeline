"""
Credit Risk Scoring & Unsupervised Segmentation Pipeline
Author: Alfredo Ramos (@Alfredo-ctrl)

Implements a leak-free scikit-learn ColumnTransformer & Pipeline architecture
for binary default classification and PCA/K-Means borrower segmentation on the
German Credit dataset.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

DATASET_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/credit.csv"


def load_and_prepare_data(url: str = DATASET_URL) -> tuple[pd.DataFrame, pd.Series]:
    """Loads the German Credit dataset and normalizes the binary target (0 = non-default, 1 = default)."""
    df = pd.read_csv(url)
    if set(df["default"].unique()) == {1, 2}:
        df["default"] = df["default"].map({1: 0, 2: 1})
    X = df.drop("default", axis=1)
    y = df["default"]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Constructs a leak-free ColumnTransformer for numerical scaling and categorical encoding."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )


def run_benchmark(output_dir: str) -> dict:
    """Executes supervised model comparison, hyperparameter optimization, and PCA/K-Means clustering."""
    os.makedirs(output_dir, exist_ok=True)
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "SVM (RBF Kernel)": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    plt.style.use("dark_background")
    fig_cm, axes_cm = plt.subplots(1, 4, figsize=(18, 4.2), facecolor="#0B1120")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6), facecolor="#0B1120")
    ax_roc.set_facecolor("#0F172A")

    palette = ["#38BDF8", "#818CF8", "#10B981", "#F59E0B"]
    metrics_summary = []

    for idx, (name, clf) in enumerate(models.items()):
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, preds))
        rec = float(recall_score(y_test, preds))
        prec = float(precision_score(y_test, preds))
        f1 = float(f1_score(y_test, preds))
        auc = float(roc_auc_score(y_test, probs))

        metrics_summary.append({
            "model": name,
            "accuracy": round(acc, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4)
        })

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        ax = axes_cm[idx]
        ax.set_facecolor("#0F172A")
        im = ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", color="#F8FAFC", fontsize=13, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Performing", "Default"], fontsize=9, color="#CBD5E1")
        ax.set_yticklabels(["Performing", "Default"], fontsize=9, color="#CBD5E1")
        ax.set_title(f"{name}\nAUC: {auc:.3f} | Recall: {rec:.3f}", fontsize=10, color="#38BDF8", pad=10)
        ax.set_xlabel("Predicted Class", fontsize=9, color="#94A3B8")
        if idx == 0:
            ax.set_ylabel("Actual Class", fontsize=9, color="#94A3B8")

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=palette[idx], linewidth=2.2)

    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=160, facecolor="#0B1120")
    plt.close(fig_cm)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#475569", linewidth=1.2, label="Random Baseline (0.500)")
    ax_roc.set_title("ROC Curve Comparison — German Credit Test Set", fontsize=12, color="#F8FAFC", pad=12)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, color="#94A3B8")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10, color="#94A3B8")
    ax_roc.grid(color="#1E293B", linestyle="--", linewidth=0.7)
    ax_roc.legend(loc="lower right", frameon=True, facecolor="#0B1120", edgecolor="#1E293B")
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=160, facecolor="#0B1120")
    plt.close(fig_roc)

    # Hyperparameter Optimization with GridSearchCV
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 5.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }
    grid_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(grid_pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    test_preds = best_model.predict(X_test)
    test_probs = best_model.predict_proba(X_test)[:, 1]

    # Unsupervised Analysis: PCA + K-Means
    X_all_prep = preprocessor.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all_prep)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_all_prep)

    fig_pca, axes_pca = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0B1120")
    for a in axes_pca:
        a.set_facecolor("#0F172A")
        a.grid(color="#1E293B", linestyle="--", linewidth=0.6)

    cluster_colors = ["#38BDF8", "#818CF8", "#10B981"]
    for c_id in range(3):
        mask = clusters == c_id
        axes_pca[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=cluster_colors[c_id], label=f"Cluster {c_id}", alpha=0.75, s=26)
    axes_pca[0].set_title("Unsupervised K-Means (K=3) in PCA Projection", fontsize=11, color="#F8FAFC")
    axes_pca[0].set_xlabel("Principal Component 1", fontsize=9, color="#94A3B8")
    axes_pca[0].set_ylabel("Principal Component 2", fontsize=9, color="#94A3B8")
    axes_pca[0].legend(facecolor="#0B1120", edgecolor="#1E293B")

    target_colors = {0: ("#38BDF8", "Performing (0)"), 1: ("#EF4444", "Default (1)")}
    for t_val, (col, lbl) in target_colors.items():
        mask = (y == t_val).values
        axes_pca[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=col, label=lbl, alpha=0.75, s=26)
    axes_pca[1].set_title("Ground-Truth Default Distribution in PCA Space", fontsize=11, color="#F8FAFC")
    axes_pca[1].set_xlabel("Principal Component 1", fontsize=9, color="#94A3B8")
    axes_pca[1].set_ylabel("Principal Component 2", fontsize=9, color="#94A3B8")
    axes_pca[1].legend(facecolor="#0B1120", edgecolor="#1E293B")

    fig_pca.tight_layout()
    fig_pca.savefig(os.path.join(output_dir, "pca_kmeans_clusters.png"), dpi=160, facecolor="#0B1120")
    plt.close(fig_pca)

    report = {
        "dataset_rows": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "models": metrics_summary,
        "grid_search_best_params": grid.best_params_,
        "grid_search_cv_roc_auc": round(float(grid.best_score_), 4),
        "tuned_test_roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
        "tuned_test_recall": round(float(recall_score(y_test, test_preds)), 4),
        "pca_explained_variance_pct": round(float(pca.explained_variance_ratio_.sum() * 100), 2)
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    summary = run_benchmark(out_dir)
    print(json.dumps(summary, indent=2))
