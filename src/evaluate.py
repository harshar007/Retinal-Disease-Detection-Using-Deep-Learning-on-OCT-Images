"""
Module: src.evaluate
Comprehensive evaluation of trained deep learning models on unseen retinal OCT test data.
Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and class-wise diagnostics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def evaluate_model(model, test_loader, device=None, class_names=None):
    """
    Evaluates a PyTorch model on a test DataLoader.
    Returns:
        dict containing metrics, confusion matrix, and prediction arrays.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = float(accuracy_score(all_targets, all_preds))
    prec_macro = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    rec_macro = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    f1_macro = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    prec_weighted = float(precision_score(all_targets, all_preds, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(all_targets, all_preds, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

    cm = confusion_matrix(all_targets, all_preds)
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)

    # Class-wise metrics
    class_metrics = {}
    for i, c_name in enumerate(class_names):
        class_metrics[c_name] = {
            "precision": float(report_dict[c_name]["precision"]),
            "recall": float(report_dict[c_name]["recall"]),
            "f1-score": float(report_dict[c_name]["f1-score"]),
            "support": int(report_dict[c_name]["support"])
        }

    # Identify misclassifications
    misclassified_indices = np.where(all_preds != all_targets)[0]
    misclassifications = []
    for idx in misclassified_indices:
        misclassifications.append({
            "index": int(idx),
            "true_label": class_names[all_targets[idx]],
            "predicted_label": class_names[all_preds[idx]],
            "confidence": float(all_probs[idx][all_preds[idx]])
        })

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report_str,
        "class_metrics": class_metrics,
        "num_test_samples": len(all_targets),
        "num_misclassified": len(misclassified_indices),
        "misclassifications": misclassifications,
        "y_true": all_targets,
        "y_pred": all_preds,
        "y_probs": all_probs
    }

    return metrics
