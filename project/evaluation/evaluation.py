"""
Jalankan seluruh pipeline evaluasi untuk model CNN pada dataset pneumonia X-ray,dimana;
- Mengumpulkan prediksi dari test_loader
- Menghitung metrik: accuracy, precision, recall, F1-score, ROC-AUC
- Plot confusion matrix, ROC curve, prediction samples, error analysis
- Generate Grad-CAM untuk sampel gambar
- Bandingkan beberapa model dalam satu tabel ringkas

"""

import os
import torch
import numpy as np
# import cv2
# from PIL import Image

from evaluation.metrics import (
    get_predictions,
    compute_metrics,
    print_metrics
)
from evaluation.gradcam import GradCAM, batch_gradcam, visualize_gradcam
from evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    plot_prediction_samples,
    plot_gradcam_grid,
    plot_error_analysis,
    compare_training_curves,
    compare_roc_curves,
    plot_metrics_comparison,
)
from models.baseline_model import BaselineCNN
from models.transfer_learning import ResNet50TransferLearning
from preprocessing.dataloader import test_loader

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
OUTPUT_DIR = "evaluation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Sung aja
# ------------------------------------------------------------------

def load_baseline(model_path="best_baseline_cnn.pth"):
    model = BaselineCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"Baseline CNN loaded from: {model_path}")
    return model

def load_transfer(model_path="resnet50_frozen.pth"):
    model = ResNet50TransferLearning(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"ResNet50 Transfer loaded from: {model_path}")
    return model

def run_full_evaluation(
    model,
    model_name="Model",
    class_names=None,
    run_gradcam=True,
    baseline_model_path="best_baseline_cnn.pth",
    transfer_model_path="resnet50_frozen.pth",
    history_baseline=None,
    history_transfer=None,
):
    if class_names is None:
        class_names = test_loader

    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"EVALUASI MODEL: {model_name}")
    print(f"{'='*60}\n")

    # Step 1: Kumpulkan prediksi
    print("Mengumpulkan prediksi pada test set ...")
    all_labels, all_preds, all_probs = get_predictions(model, class_names)

    # Step 2: Metrik 
    print("\nMenghitung metrik ...")
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print_metrics(metrics, model_name)

    # Step 3: Training curves
    if history is not None:
        print("\nPlotting training curves ...")
        plot_training_curves(history, model_name=model_name, save_path=os.path.join(OUTPUT_DIR, f"{model_name}_training_curves.png")) 

    # Confusion Matrix
    print("\nPlotting confusion matrix ...")
    plot_confusion_matrix(metrics["confusion_matrix"], name=CLASS_NAMES, model_name=model_name, save_path=os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"))

    # Step 5: Roc curve
    print("\nPlotting ROC curve ...")
    if "auc" in metrics:
        plot_roc_curve(all_labels, all_probs, model_name=model_name, save_path=os.path.join(OUTPUT_DIR, f"roc_curve_{model_name.lower().replace(' ', '_')}.png"))

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

    # with torch.no_grad():
    #     out = model(sample_images.to(device))
    #     sample_preds = torch.argmax(torch.softmax(out, dim=1), dim=1).cpu().numpy()
    #     sample_probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

    # Konversi tensor images ke numpy untuk plotting
    images_np = sample_images.numpy() 

    plot_prediction_samples(
        model, class_names, n=8, name=CLASS_NAMES, save_path=os.path.join(OUTPUT_DIR, f"pred_samples_{model_name.lower().replace(' ', '_')}.png")
    )

    print("\nPlotting error analysis ...")
    # all_images_list = []
    # for imgs, _ in test_loader:
    #     all_images_list.append(imgs)
    # all_images_np = torch.cat(all_images_list, dim=0).numpy()

    plot_error_analysis(
        model, name=CLASS_NAMES, n_fp=4, n_fn=4, save_path=os.path.join(OUTPUT_DIR, f"error_analysis_{model_name.lower().replace(' ', '_')}.png")
    )

    if run_gradcam:
        _run_gradcam(model, model_name, class_names)

    print(f"\nEvaluasi {model_name} selesai!\n")
    return metrics


# ------------------------------------------------------------------
# 2. GMilih Target Layer untuk Grad-CAM & Generate Grad-CAM
# ------------------------------------------------------------------

def _run_gradcam(model, model_name, class_names):
    print("\nRunning Grad-CAM ...")
    try:
        if hasattr(model, "model") and hasattr(model.model, "layer4"):
            target_layer = model.model.layer4[-1]
        elif hasattr(model, "conv3"):
            target_layer = model.conv3
        else:
            print("Grad-CAM: target layer tidak ditemukan, dilewati.")
            return
        
        print(f"\n[Grad-CAM] Generating for {model_name} ...")
        batch_gradcam(model, target_layer, class_names, n_samples=6, names=CLASS_NAMES, save_path=os.path.join(OUTPUT_DIR, f"gradcam_{model_name.lower().replace(' ', '_')}.png"))
    except Exception as e:
        print(f"Grad-CAM gagal: {e}")

# ------------------------------------------------------------------
# 3. Perbandingan model side-by-side
# ------------------------------------------------------------------

def compare_models(
        baseline_model_path="best_baseline_cnn.pth",
        transfer_model_path="resnet50_frozen.pth",
        history_baseline=None,
        history_transfer=None,
        class_names=None,
    ):

    if class_names is None:
        class_names = test_loader
    
    # 1. Evaluasi model baseline
    baseline_model = load_baseline(baseline_model_path)
    baseline_metrics = run_full_evaluation(
        baseline_model,
        model_name="Baseline CNN",
        class_names=class_names,
        history=history_baseline,
        run_gradcam=True,
    )
    # 2. Evaluasi model transfer learning
    transfer_model = load_transfer(transfer_model_path)
    transfer_metrics = run_full_evaluation(
        transfer_model,
        model_name="ResNet50 Transfer",
        class_names=class_names,
        history=history_transfer,
        run_gradcam=True,
    )   

    # 3. Bandingkan kedua model
    print("\n" + "="*65)
    print("MODEL COMPARISON")
    print("="*65)
    
    plot_metrics_comparison(
        {"Baseline CNN": baseline_metrics,
         "ResNet50 Transfer": transfer_metrics
        },
        save_path=os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    )
    if history_baseline is not None and history_transfer is not None:
        compare_training_curves(
            history_baseline, history_transfer,
            save_path=os.path.join(OUTPUT_DIR, "training_curves_comparison.png")
        )
    if "auc" in baseline_metrics and "auc" in transfer_metrics:
        preds_b, labels_b, probs_b = get_predictions(baseline_model, class_names)
        preds_t, labels_t, probs_t = get_predictions(transfer_model, class_names)
        compare_roc_curves(
            {
                "Baseline CNN":      {"labels": labels_b, "probs": probs_b},
                "ResNet50 Transfer": {"labels": labels_t, "probs": probs_t},
            },
            save_path=os.path.join(OUTPUT_DIR, "roc_comparison.png")
        )
    
    print(f"\nSemua output disimpan di folder: '{OUTPUT_DIR}/'")
    print(f"    Metrics Baseline : Acc={baseline_metrics['accuracy']:.4f} | "
          f"F1={baseline_metrics['f1']:.4f} | AUC={baseline_metrics.get('auc', 'N/A')}")
    print(f"    Metrics Transfer  : Acc={transfer_metrics['accuracy']:.4f} | "
          f"F1={transfer_metrics['f1']:.4f} | AUC={transfer_metrics.get('auc', 'N/A')}")

    return baseline_metrics, transfer_metrics

# ------------------------------------------------------------------
# 3. Visualisasi standalone Grd-Cam
# ------------------------------------------------------------------
def inspect_single_image(model, input_tensor, label,
                          model_type="resnet", save_path=None):
    if model_type == "resnet":
        target_layer = model.model.layer4[-1]
    else:
        target_layer = model.conv3

    visualize_gradcam(
        model, target_layer, input_tensor, label,
        name=CLASS_NAMES,
        save_path=save_path
    )
