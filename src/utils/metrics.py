# Utility functions for computing metrics (accuracy, confusion matrix, etc.)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate various metrics for classification

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        dict: Dictionary containing calculated metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    return metrics


def print_metrics(metrics, class_names=None):
    """
    Print metrics in a readable format

    Args:
        metrics: Dictionary containing metrics
        class_names: List of class names
    """
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    if class_names is not None:
        print("\nConfusion Matrix:")
        conf_matrix = metrics["confusion_matrix"]
        # Print header
        print("Predicted →", end="")
        for name in class_names:
            print(f"\t{name}", end="")
        print("\nActual ↓")

        # Print rows
        for i, name in enumerate(class_names):
            print(f"{name}", end="")
            for j in range(len(class_names)):
                print(f"\t{conf_matrix[i, j]}", end="")
            print()
