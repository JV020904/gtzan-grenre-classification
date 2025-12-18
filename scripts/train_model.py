"""
Author: Jose Varela
Email: jvarela@haverford.edu
Train classical ML models on extracted GTZAN audio features.
Generates:
- Trained models
- Confusion matrix plots
- Accuracy comparison bar chart
- Metrics summary JSON
"""

import os
import json
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from config import FEATURES_DIR, MODEL_DIR, RESULTS_DIR


# Helper function: Confusion matrix plotting
def plot_confusion_matrix(y_true, y_pred, model_name, labels, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    #Plot the title, axis, and save the figure
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_{model_name}.png"))
    plt.close()


# Loading the features
def load_features():
    X = np.load(os.path.join(FEATURES_DIR, "X.npy"))
    y = np.load(os.path.join(FEATURES_DIR, "y.npy"))

    with open(os.path.join(FEATURES_DIR, "labels.json"), "r") as f:
        labels = json.load(f)

    return X, y, labels


# Main function for training
def main():
    print("Loading features...")
    X, y, LABELS = load_features()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    metrics_summary = {}

    # 1. Logistic Regression
    print("\n=== Logistic Regression ===")
    logreg_clf = LogisticRegression(max_iter=5000, n_jobs=-1)
    logreg_clf.fit(X_train, y_train)

    y_pred_lr = logreg_clf.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    print(f"Accuracy: {acc_lr:.3f}")
    print(classification_report(y_test, y_pred_lr))

    joblib.dump(logreg_clf, os.path.join(MODEL_DIR, "logreg.joblib"))
    plot_confusion_matrix(y_test, y_pred_lr, "logreg", LABELS)

    metrics_summary["logreg"] = {"accuracy": acc_lr}


    # 2. Random Forest
    print("\n=== Random Forest ===")
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)

    y_pred_rf = rf_clf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print(f"Accuracy: {acc_rf:.3f}")
    print(classification_report(y_test, y_pred_rf))

    joblib.dump(rf_clf, os.path.join(MODEL_DIR, "rf.joblib"))
    plot_confusion_matrix(y_test, y_pred_rf, "random_forest", LABELS)

    metrics_summary["random_forest"] = {"accuracy": acc_rf}


    # 3. MLP Classifier
    print("\n=== MLP Classifier ===")
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=64,
        max_iter=500,
        random_state=42
    )
    mlp_clf.fit(X_train, y_train)

    y_pred_mlp = mlp_clf.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)

    print(f"Accuracy: {acc_mlp:.3f}")
    print(classification_report(y_test, y_pred_mlp))

    joblib.dump(mlp_clf, os.path.join(MODEL_DIR, "mlp.joblib"))
    plot_confusion_matrix(y_test, y_pred_mlp, "mlp", LABELS)

    metrics_summary["mlp"] = {"accuracy": acc_mlp}


    # Accuracy comparison bar chart
    model_names = ["LogReg", "RandomForest", "MLP"]
    accuracies = [acc_lr, acc_rf, acc_mlp]

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, accuracies)
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison on the GTZAN Dataset")
    plt.tight_layout()
    plt.savefig("plots/accuracy_comparison.png")
    plt.close()


    # Saving the metrics summary
    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print("\nSaved all models, plots, and metric summary.")


if __name__ == "__main__":
    main()
