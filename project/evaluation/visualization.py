"""
Ini buat visualisasi hasil evaluasi model klasifikasi pneumonia pada X-ray. Plot-plotnya
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import itertools
from sklearn.metrics import roc_curve, auc
import torch

# ------------------------------------------------------------------
# Global theme — consistent across all plots
# ------------------------------------------------------------------

PALETTE = {
    "blue":   "#2C7BE5", 
    "red":    "#E5392C",
    "green":  "#27AE60",
    "orange": "#F39C12",
    "gray":   "#7F8C8D",
    "light":  "#F8F9FA",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["light"],
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titleweight":  "bold",
})


def _style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    
# ------------------------------------------------------------------
# 1. Confusion Matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(cm, class_names=None, model_name="Model", normalize=False, save_path=None):
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
        title = f"Normalized Confusion Matrix — {model_name}"
    else:
        cm_display = cm
        fmt = "d"
        title = f"Confusion Matrix — {model_name}"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ------------------------------------------------------------------
# 2. ROC Curve
# ------------------------------------------------------------------

def plot_roc_curve(labels, probs, model_name="Model", save_path=None):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE["blue"], lw=2,
            label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["gray"], lw=1.5,
            linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["blue"])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    _style_ax(ax, f"ROC Curve — {model_name}", "False Positive Rate", "True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return roc_auc

def compare_roc_curves(results_dict, save_path=None):
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, data) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(data["labels"], data["probs"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], color=PALETTE["gray"], lw=1.5,
            linestyle="--", label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    _style_ax(ax, "ROC Curve Comparison", "False Positive Rate", "True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()

# ------------------------------------------------------------------
# 3. Training Curves
# ------------------------------------------------------------------

def plot_training_curves(history: dict, model_name="Model", save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History — {model_name}", fontsize=14, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], color=PALETTE["blue"],  lw=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   color=PALETTE["red"],   lw=2, label="Val Loss", linestyle="--")
    _style_ax(ax1, "Loss per Epoch", "Epoch", "Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, history["train_acc"], color=PALETTE["blue"],  lw=2, label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   color=PALETTE["red"],   lw=2, label="Val Acc", linestyle="--")
    ax2.set_ylim([0, 1.05])
    _style_ax(ax2, "Accuracy per Epoch", "Epoch", "Accuracy")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()

def compare_training_curves(history_baseline, history_transfer, save_path=None):
    epochs_b = range(1, len(history_baseline["val_acc"]) + 1)
    epochs_t = range(1, len(history_transfer["val_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison — Baseline CNN vs Transfer Learning", fontsize=13, fontweight="bold")

    labels = [("Baseline CNN", PALETTE["blue"]), 
              ("ResNet50 Transfer", PALETTE["orange"])]

    for ax, key, title, ylim in [
        (axes[0], "val_loss",  "Validation Loss",     None),
        (axes[1], "val_acc",   "Validation Accuracy", (0, 1.05)),
    ]:
        ax.plot(epochs_b, history_baseline[key], color=labels[0][1], lw=2, label=labels[0][0])
        ax.plot(epochs_t, history_transfer[key], color=labels[1][1], lw=2, label=labels[1][0], linestyle="--")
        _style_ax(ax, title, "Epoch", key.split("_")[1].capitalize())
        if ylim:
            ax.set_ylim(ylim)
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()

# ------------------------------------------------------------------
# 4. Prediction Samples
# ------------------------------------------------------------------
def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    t = tensor.clone().cpu().numpy().transpose(1, 2, 0)
    t = t * np.array(std) + np.array(mean)
    return np.clip(t, 0, 1)

def plot_prediction_samples(
    model, data_loader, n=8,
    class_names=None, save_path=None
):
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    device = next(model.parameters()).device
    model.eval()

    images, labels, preds = [], [], []
    with torch.no_grad():
        for imgs, lbls in data_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            pred = outputs.argmax(dim=1).cpu()
            images.extend(imgs.cpu())
            labels.extend(lbls.numpy())
            preds.extend(pred.numpy())
            if len(images) >= n:
                break

    images, labels, preds = images[:n], labels[:n], preds[:n]
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (img, lbl, pred) in enumerate(zip(images, labels, preds)):
        ax = axes[i]
        ax.imshow(denormalize(img))
        correct = pred == lbl
        border_color = PALETTE["green"] if correct else PALETTE["red"]
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        ax.set_title(
            f"True: {class_names[lbl]}\nPred: {class_names[pred]}",
            fontsize=9,
            color="black"
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Prediction Examples  (green = correct, red = wrong)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ------------------------------------------------------------------
# 5. Error Analysis
# ------------------------------------------------------------------

def plot_error_analysis(
    model, data_loader, class_names=None,
    n_fp=4, n_fn=4, save_path=None
):
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    device = next(model.parameters()).device
    model.eval()

    fp_images, fn_images = [], []

    with torch.no_grad():
        for imgs, lbls in data_loader:
            imgs_dev = imgs.to(device)
            outputs = model(imgs_dev)
            preds = outputs.argmax(dim=1).cpu().numpy()

            for img, lbl, pred in zip(imgs.cpu(), lbls.numpy(), preds):
                lbl = int(lbl)
                pred = int(pred)
                if lbl == 0 and pred == 1 and len(fp_images) < n_fp:
                    fp_images.append(img)
                elif lbl == 1 and pred == 0 and len(fn_images) < n_fn:
                    fn_images.append(img)

            if len(fp_images) >= n_fp and len(fn_images) >= n_fn:
                break

    n_cols = max(n_fp, n_fn)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

    row_titles = [
        f"False Positives ({n_fp})  — NORMAL predicted as PNEUMONIA",
        f"False Negatives ({n_fn})  — PNEUMONIA predicted as NORMAL",
    ]
    sets = [fp_images, fn_images]
    colors = [PALETTE["orange"], PALETTE["red"]]

    for row, (title, imgs, color) in enumerate(zip(row_titles, sets, colors)):
        for col in range(n_cols):
            ax = axes[row][col]
            if col < len(imgs):
                ax.imshow(denormalize(imgs[col]))
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row][0].set_ylabel(title, fontsize=9, fontweight="bold",
                                rotation=90, labelpad=8)

    fig.suptitle("Error Analysis — False Positives & False Negatives", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()

def plot_metrics_comparison(metrics_dict, save_path=None):
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    if any("auc" in v for v in metrics_dict.values()):
        metric_keys.append("auc")

    model_names = list(metrics_dict.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / len(model_names)
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, metrics) in enumerate(metrics_dict.items()):
        vals = [metrics.get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width - (len(model_names) - 1) * width / 2,
                      vals, width * 0.9, label=name, color=colors[i % len(colors)],
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in metric_keys])
    ax.set_ylim([0, 1.15])
    _style_ax(ax, "Model Performance Comparison", ylabel="Score")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()