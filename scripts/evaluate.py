"""
Author: Jose Varela
Email: jvarela@haverford.edu
This file is for evaluating the trained models
"""
import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_DIR = "features"
MODEL_DIR = "models"
RESULTS_DIR = "results"


def main():
    print("Loading features...")
    X = np.load(os.path.join(FEATURES_DIR, "X.npy"))
    y = np.load(os.path.join(FEATURES_DIR, "y.npy"))

    with open(os.path.join(FEATURES_DIR, "labels.json"), "r") as f:
        LABELS = json.load(f)

    # IMPORTANT: same split as training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Loading model...")
    clf = joblib.load(os.path.join(MODEL_DIR, "rf.joblib"))
    model_name = "Random Forest"

    print("Running evaluation on test set...")
    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 8))
    #Making a heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix – Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_rf_.png"))
    plt.close()

    print("Saved confusion matrix to results/")


if __name__ == "__main__":
    main()
