"""
Jalankan seluruh pipeline evaluasi untuk model CNN pada dataset pneumonia X-ray,dimana;
- Mengumpulkan prediksi dari test_loader
- Menghitung metrik: accuracy, precision, recall, F1-score, ROC-AUC
- Plot confusion matrix, ROC curve, prediction samples, error analysis
- Generate Grad-CAM untuk sampel gambar
- Bandingkan beberapa model dalam satu tabel ;9

"""

import os
import torch
import numpy as np
import cv2
from PIL import Image

from evaluation.metrics import (
    get_all_predictions,
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_auc,
)
from evaluation.gradcam import GradCAM, get_target_layer
from evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    plot_prediction_samples,
    plot_gradcam_grid,
    plot_error_analysis,
)


# ------------------------------------------------------------------
# 1. Pipeline evaluasi lengkap
# ------------------------------------------------------------------

def run_full_evaluation(
    model,
    test_loader,
    model_name="Model",
    class_names=None,
    save_dir=None,
    device=None,
):
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"EVALUASI MODEL: {model_name}")
    print(f"{'='*60}\n")

    # Step 1: Kumpulkan prediksi
    print("Mengumpulkan prediksi pada test set ...")
    all_labels, all_preds, all_probs = get_all_predictions(model, test_loader, device)

    # Step 2: Metrik 
    print("\nMenghitung metrik ...")
    metrics = compute_metrics(all_labels, all_preds, class_names)
    cm      = compute_confusion_matrix(all_labels, all_preds)
    fpr, tpr, _, roc_auc = compute_roc_auc(all_labels, all_probs)

    # Step 3: Plot 
    _save = lambda name: os.path.join(save_dir, name) if save_dir else None

    print("\nPlotting confusion matrix ...")
    plot_confusion_matrix(cm, class_names, save_path=_save(f"{model_name}_cm.png"))

    print("\nPlotting ROC curve ...")
    plot_roc_curve(fpr, tpr, roc_auc, save_path=_save(f"{model_name}_roc.png"))

    # Step 4: Sample gambar dari test_loader untuk visualisasi
    print("\nPlotting prediction samples ...")
    sample_images, sample_labels_raw = [], []
    for imgs, lbls in test_loader:
        sample_images.append(imgs)
        sample_labels_raw.append(lbls)
        if sum(len(b) for b in sample_images) >= 16:
            break

    sample_images = torch.cat(sample_images, dim=0)[:16]
    sample_labels = torch.cat(sample_labels_raw, dim=0)[:16].numpy()

    with torch.no_grad():
        out = model(sample_images.to(device))
        sample_preds = torch.argmax(torch.softmax(out, dim=1), dim=1).cpu().numpy()
        sample_probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

    # Konversi tensor images ke numpy untuk plotting
    images_np = sample_images.numpy() 

    plot_prediction_samples(
        images_np, sample_labels, sample_preds, sample_probs,
        class_names=class_names, n=16,
        save_path=_save(f"{model_name}_samples.png"),
    )

    print("\nPlotting error analysis ...")
    all_images_list = []
    for imgs, _ in test_loader:
        all_images_list.append(imgs)
    all_images_np = torch.cat(all_images_list, dim=0).numpy()

    plot_error_analysis(
        all_images_np, all_labels, all_preds, all_probs,
        class_names=class_names, n_each=8,
        save_path=_save(f"{model_name}_errors.png"),
    )

    results = {
        "metrics":  metrics,
        "cm":       cm,
        "fpr":      fpr,
        "tpr":      tpr,
        "roc_auc":  roc_auc,
        "labels":   all_labels,
        "preds":    all_preds,
        "probs":    all_probs,
    }

    print(f"\nEvaluasi {model_name} selesai!\n")
    return results


# ------------------------------------------------------------------
# 2. Grad-CAM untuk sekumpulan sampel
# ------------------------------------------------------------------

def evaluate_gradcam(
    model,
    test_loader,
    n_samples=6,
    class_names=None,
    device=None,
    save_path=None,
    target_layer=None,
):
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect target layer
    if target_layer is None:
        target_layer = get_target_layer(model)
        print(f"[GradCAM] Target layer: {target_layer.__class__.__name__}")

    gcam = GradCAM(model, target_layer)

    gradcam_samples = []
    collected = 0

    for imgs, lbls in test_loader:
        for idx in range(len(imgs)):
            if collected >= n_samples:
                break

            input_tensor = imgs[idx].unsqueeze(0)
            true_label   = int(lbls[idx].item())

            heatmap, pred_label, pred_prob = gcam.generate(
                input_tensor, class_idx=None, device=device
            )

            img_np = imgs[idx].numpy() 
            # Denormalize if  needed with assumption that mean and std is 0.5
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = np.clip(img_np, 0, 1)
            img_uint8 = (img_np * 255).astype(np.uint8)

            overlay_bgr, _ = GradCAM.overlay(img_uint8, heatmap)

            gradcam_samples.append({
                "image":      img_uint8,
                "heatmap":    heatmap,
                "overlay":    overlay_bgr,
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_prob":  pred_prob,
            })
            collected += 1

        if collected >= n_samples:
            break

    gcam.remove_hooks()

    print(f"\nPlotting Grad-CAM untuk {len(gradcam_samples)} sampel ...")
    plot_gradcam_grid(
        gradcam_samples,
        class_names=class_names,
        title="Grad-CAM — Interpretasi Keputusan Model",
        save_path=save_path,
    )

    return gradcam_samples


# ------------------------------------------------------------------
# 3. Perbandingan model side-by-side
# ------------------------------------------------------------------

def compare_models(results_dict: dict):
    import pandas as pd

    rows = []
    for name, res in results_dict.items():
        m = res["metrics"]
        rows.append({
            "Model":     name,
            "Accuracy":  f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall":    f"{m['recall']:.4f}",
            "F1-Score":  f"{m['f1_score']:.4f}",
            "ROC-AUC":   f"{res['roc_auc']:.4f}",
        })

    df = pd.DataFrame(rows)
    print("\n" + "="*65)
    print("MODEL COMPARISON")
    print("="*65)
    print(df.to_string(index=False))
    print("="*65)
    return df
