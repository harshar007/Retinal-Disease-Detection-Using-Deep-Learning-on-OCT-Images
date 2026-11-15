"""
Module: src.utils
Utility functions for serialization (saving/loading models and results to .pkl format),
plotting evaluation metrics, confusion matrices, and Grad-CAM visualizations.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch


def save_pickle(obj, filepath):
    """Saves any Python object to a .pkl file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Successfully serialized to PKL: {filepath}")


def load_pickle(filepath):
    """Loads a Python object from a .pkl file."""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_model_package_pkl(model, model_name, metrics, filepath, extra_info=None):
    """
    Saves complete model package into .pkl format as required:
    - architecture name
    - model weights (state_dict)
    - model object
    - test performance metrics (accuracy, precision, recall, f1)
    - metadata (classes, input shape, normalization)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Remove any registered hooks so the PyTorch model object can be pickled cleanly
    for m in model.modules():
        if hasattr(m, "_forward_hooks"):
            m._forward_hooks.clear()
        if hasattr(m, "_backward_hooks"):
            m._backward_hooks.clear()
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()
        if hasattr(m, "_full_backward_hooks"):
            m._full_backward_hooks.clear()

    package = {
        "model_name": model_name,
        "model_architecture": str(model.__class__.__name__),
        "state_dict": model.state_dict(),
        "model": model,  # complete model object
        "metrics": metrics,
        "class_names": ["CNV", "DME", "DRUSEN", "NORMAL"],
        "input_size": (3, 224, 224),
        "extra_info": extra_info or {}
    }
    with open(filepath, "wb") as f:
        pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model package saved in PKL format: {filepath}")


def plot_confusion_matrices(cm_dict, class_names, save_path="outputs/confusion_matrices.png"):
    """
    Plots confusion matrices for multiple models in a grid and saves to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_models = len(cm_dict)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (m_name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"{m_name.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    # Hide unused subplots
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices figure to: {save_path}")


def plot_training_curves(histories, save_path="outputs/training_curves.png"):
    """
    Plots Train/Val Loss and Train/Val Accuracy curves for all trained models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for m_name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax1.plot(epochs, hist["train_loss"], label=f"{m_name} (Train)", linestyle="--")
        ax1.plot(epochs, hist["val_loss"], label=f"{m_name} (Val)")
        
        ax2.plot(epochs, hist["train_acc"], label=f"{m_name} (Train)", linestyle="--")
        ax2.plot(epochs, hist["val_acc"], label=f"{m_name} (Val)")

    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=10)
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves to: {save_path}")


def plot_model_comparison_bar(summary_df, save_path="outputs/model_comparison_metrics.png"):
    """
    Plots a grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = summary_df["Model"].tolist()
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    x = np.arange(len(models))
    width = 0.2
    
    colors = ["#2b5c8f", "#388e3c", "#e65100", "#7b1fa2"]
    for i, metric in enumerate(metrics):
        values = [summary_df.loc[summary_df["Model"] == m, metric].values[0] for m in models]
        rects = ax.bar(x + (i - 1.5) * width, values, width, label=metric, color=colors[i], alpha=0.9)
        # Add labels above bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Comprehensive Model Benchmark Comparison (Test Set)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved model comparison chart to: {save_path}")
