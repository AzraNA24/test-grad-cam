"""
Ini buat visualisasi hasil evaluasi model klasifikasi pneumonia pada X-ray. Plot-plotnya
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import itertools


# ------------------------------------------------------------------
# Tema global — konsisten di semua plot
# ------------------------------------------------------------------

PALETTE = {
    "primary":   "#2C7BE5",
    "danger":    "#E5392C",
    "success":   "#27AE60",
    "warning":   "#F39C12",
    "neutral":   "#7F8C8D",
    "bg":        "#F8F9FA",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titleweight":  "bold",
})


# ------------------------------------------------------------------
# 1. Confusion Matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(cm, class_names=None, save_path=None):
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Proportion"},
    )

    # Anotasi gabungan nilai & persentase
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j + 0.5, i + 0.5,
            f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color="white" if cm_norm[i, j] > 0.5 else "black",
        )

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names, rotation=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix disimpan: {save_path}")
    plt.show()
    return fig


# ------------------------------------------------------------------
# 2. ROC Curve
# ------------------------------------------------------------------

def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1.5,
            linestyle="--", label="Random Classifier")

    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["primary"])

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve disimpan: {save_path}")
    plt.show()
    return fig


# ------------------------------------------------------------------
# 3. Training Curves
# ------------------------------------------------------------------

def plot_training_curves(history: dict, model_name="Model", save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=13, fontweight="bold")

    # --- Loss ---
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=PALETTE["primary"],
            lw=2, label="Train Loss", marker="o", markersize=4)
    ax.plot(epochs, history["val_loss"], color=PALETTE["danger"],
            lw=2, label="Val Loss", marker="s", markersize=4, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per Epoch")
    ax.legend()

    # --- Accuracy ---
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], color=PALETTE["primary"],
            lw=2, label="Train Acc", marker="o", markersize=4)
    ax.plot(epochs, history["val_acc"], color=PALETTE["success"],
            lw=2, label="Val Acc", marker="s", markersize=4, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per Epoch")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves disimpan: {save_path}")
    plt.show()
    return fig


# ------------------------------------------------------------------
# 4. Prediction Samples
# ------------------------------------------------------------------

def plot_prediction_samples(
    images, labels, preds, probs,
    class_names=None, n=16, save_path=None
):
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    n = min(n, len(images))
    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        ax = axes[i]
        img = images[i]

        # Normalisasi ke 0-1 untuk imshow
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32) / 255.0
        img = np.clip(img, 0, 1)

        if img.ndim == 2 or (img.ndim == 3 and img.shape[0] in [1, 3]):
            # Channel-first → channel-last
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]

        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

        correct = (labels[i] == preds[i])
        border_color = PALETTE["success"] if correct else PALETTE["danger"]
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3.5)

        ax.set_title(
            f"GT: {class_names[labels[i]]}\n"
            f"Pred: {class_names[preds[i]]} ({probs[i]:.2f})",
            fontsize=8,
            color=border_color,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Sembunyikan axes kosong
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Prediction Samples", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Prediction samples disimpan: {save_path}")
    plt.show()
    return fig


# ------------------------------------------------------------------
# 5. Grad-CAM Grid
# ------------------------------------------------------------------

def plot_gradcam_grid(
    samples: list,
    class_names=None,
    title="Grad-CAM Visualization",
    save_path=None
):
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    n = len(samples)
    cols_per_sample = 3   # Original | Heatmap | Overlay
    fig_width = cols_per_sample * n * 2.8
    fig_height = 4.0

    fig, axes = plt.subplots(1, n * cols_per_sample, figsize=(fig_width, fig_height))
    if n * cols_per_sample == 1:
        axes = [axes]

    col_labels = ["Original", "Heatmap", "Overlay"]

    for i, sample in enumerate(samples):
        img     = sample["image"]
        heatmap = sample["heatmap"]
        overlay = sample["overlay"]
        gt      = sample["true_label"]
        pred    = sample["pred_label"]
        prob    = sample["pred_prob"]

        # Normalize image untuk imshow
        if img.dtype != np.float32 and img.dtype != np.float64:
            img_show = img.astype(np.float32) / 255.0
        else:
            img_show = img.copy()
        if img_show.ndim == 3 and img_show.shape[0] in [1, 3]:
            img_show = np.transpose(img_show, (1, 2, 0))
        if img_show.ndim == 3 and img_show.shape[2] == 1:
            img_show = img_show[:, :, 0]

        visuals = [img_show, heatmap, overlay[:, :, ::-1] / 255.0]
        cmaps   = ["gray" if img_show.ndim == 2 else None, "jet", None]

        correct = (gt == pred)
        title_color = PALETTE["success"] if correct else PALETTE["danger"]

        for j, (vis, cmap) in enumerate(zip(visuals, cmaps)):
            ax = axes[i * cols_per_sample + j]
            ax.imshow(vis, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(col_labels[j], fontsize=9, labelpad=3)

            if j == 1:
                ax.set_title(
                    f"GT: {class_names[gt]} | Pred: {class_names[pred]} ({prob:.2f})",
                    fontsize=8.5,
                    color=title_color,
                    fontweight="bold",
                )

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM grid disimpan: {save_path}")
    plt.show()
    return fig


# ------------------------------------------------------------------
# 6. Error Analysis
# ------------------------------------------------------------------

def plot_error_analysis(
    images, labels, preds, probs,
    class_names=None, n_each=8, save_path=None
):
    """
    Tampilkan False Positive dan False Negative secara terpisah.

    Parameters
    ----------
    images, labels, preds, probs : sama seperti plot_prediction_samples()
    n_each    : int — jumlah error per kategori yang ditampilkan
    save_path : str opsional
    """
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    images = np.array(images)
    labels = np.array(labels)
    preds  = np.array(preds)
    probs  = np.array(probs)

    # False Positive: prediksi PNEUMONIA padahal NORMAL (label=0, pred=1)
    fp_idx = np.where((labels == 0) & (preds == 1))[0]
    # False Negative: prediksi NORMAL padahal PNEUMONIA (label=1, pred=0)
    fn_idx = np.where((labels == 1) & (preds == 0))[0]

    print(f"Total False Positives (FP): {len(fp_idx)}")
    print(f"Total False Negatives (FN): {len(fn_idx)}")

    for error_name, idx in [("False Positive (FP)", fp_idx), ("False Negative (FN)", fn_idx)]:
        if len(idx) == 0:
            print(f"Tidak ada {error_name}.")
            continue

        sel = idx[:n_each]
        n   = len(sel)
        cols = min(n, 4)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        fig.suptitle(f"Error Analysis — {error_name}", fontsize=12,
                     fontweight="bold", color=PALETTE["danger"])

        for k, ax in enumerate(axes[:n]):
            img = images[sel[k]]
            if img.dtype != np.float32 and img.dtype != np.float64:
                img = img.astype(np.float32) / 255.0
            img = np.clip(img, 0, 1)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]

            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(
                f"GT: {class_names[labels[sel[k]]]}\n"
                f"Pred: {class_names[preds[sel[k]]]} ({probs[sel[k]]:.2f})",
                fontsize=8, color=PALETTE["danger"]
            )
            ax.set_xticks([])
            ax.set_yticks([])

        for k in range(n, len(axes)):
            axes[k].set_visible(False)

        plt.tight_layout()
        if save_path:
            suffix = "fp" if "Positive" in error_name else "fn"
            path = save_path.replace(".png", f"_{suffix}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Error analysis disimpan: {path}")
        plt.show()
