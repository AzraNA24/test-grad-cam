"""
Kode mengevaluasi model pada test set dan menghitung metrik seperti Accuracy, Precision, Recall, F1-score, Confusion Matrix, dan ROC AUC.
Juga menghasilkan visualisasi seperti confusion matrix, ROC curve, dan Grad-CAM untuk sampel gambar.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)


def get_all_predictions(model, dataloader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)                         # (B, num_classes)
            probs = torch.softmax(outputs, dim=1)           # (B, num_classes)
            preds = torch.argmax(probs, dim=1)              # (B,)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())     # prob kelas positif

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_metrics(all_labels, all_preds, class_names=None):
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names if class_names else ["NORMAL", "PNEUMONIA"],
        zero_division=0,
    )

    metrics = {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1_score":  f1,
        "report":    report,
    }

    print("=" * 50)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("=" * 50)
    print("\nClassification Report:")
    print(report)

    return metrics


def compute_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    return cm


def compute_roc_auc(all_labels, all_probs):
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    return fpr, tpr, thresholds, roc_auc
