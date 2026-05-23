"""
Kode mengevaluasi model pada test set dan menghitung metrik seperti Accuracy, Precision, Recall, F1-score, Confusion Matrix, dan ROC AUC.
Juga menghasilkan visualisasi seperti confusion matrix, ROC curve, dan Grad-CAM untuk sampel gambar.
"""

import torch
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score
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
            probs = torch.softmax(outputs, dim=1)[:, 1]           # (B, num_classes)
            _,preds = torch.max(outputs, 1)             # (B,)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())    

    return (np.array(all_labels), np.array(all_preds), np.array(all_probs),)


def compute_metrics(all_labels, all_preds, all_probs=None):
    metrics = {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "classification_report": classification_report(
            all_labels, all_preds,
            target_names=["NORMAL", "PNEUMONIA"],
            zero_division=0
        ),
    }

    if all_probs is not None:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
    return metrics


# def compute_confusion_matrix(all_labels, all_preds):
#     cm = confusion_matrix(all_labels, all_preds)
#     tn, fp, fn, tp = cm.ravel()
#     print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
#     return cm


# def compute_roc_auc(all_labels, all_probs):
#     fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
#     roc_auc = auc(fpr, tpr)
#     print(f"ROC AUC: {roc_auc:.4f}")
#     return fpr, tpr, thresholds, roc_auc

def print_metrics(metrics, model_name="Model"):
    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS — {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    if "auc" in metrics:
        print(f"  AUC-ROC   : {metrics['auc']:.4f}")
    print(f"\n{metrics['classification_report']}")