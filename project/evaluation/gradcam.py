"""
Implementasi Grad-CAM untuk interpretasi keputusan model CNN pada dataset pneumonia X-ray.
Ia bekerja dengan cara mengidentifikasi area penting pada gambar yang mempengaruhi prediksi model, sehingga membantu kita memahami "kenapa" model membuat keputusan tertentu.
Dia nyimpen  feature map dan gradient dari layer konvolusi terakhir, lalu menggabungkannya untuk menghasilkan heatmap yang menunjukkan area penting pada gambar.
Lalu heatmap ini di-overlay ke gambar asli.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Pasang hook
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _save_activation(self, module, input, output):
        self.activations = output.detach()      

    def _save_gradient(self, module, grad_input, grad_output):
        """Simpan gradient feature map saat backward pass."""
        self.gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Grad-CAM utama
    # ------------------------------------------------------------------

    def generate(self, input_tensor, class_idx=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.eval()
        self.model.to(device)
        input_tensor = input_tensor.to(device)

        # Forward pass
        output = self.model(input_tensor)          # (1, num_classes)
        probs = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())

        pred_prob = float(probs[0, class_idx].item())

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Global Average Pooling pada gradients 
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)      

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        # ReLU ambil hanya kontribusi positif
        cam = F.relu(cam)

        # Resize ke ukuran input asli ~~ Change
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )         # (H, W)

        # Normalize 0–1
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam

    # ------------------------------------------------------------------
    # Overlay heatmap ke gambar asli
    # ------------------------------------------------------------------

    def overlay(self, original_image_np, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = (alpha * heatmap_rgb + (1 - alpha) * original_image_np).astype(np.uint8)
        return overlay

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self):
        """Hapus semua hooks. Panggil setelah selesai pakai GradCAM."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ------------------------------------------------------------------
# Helper: auto-detect target layer untuk model umum
# ------------------------------------------------------------------

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    t = tensor.clone().cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
    t = t * np.array(std) + np.array(mean)
    t = np.clip(t, 0, 1)
    return (t * 255).astype(np.uint8)


def visualize_gradcam(model, target_layer, input_tensor, label, class_names=None,
                      target_class=None, save_path=None, ax=None):
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, class_idx=target_class)
    gradcam.remove_hooks()

    original = denormalize(input_tensor.squeeze(0))
    overlaid = gradcam.overlay(original, heatmap)

    # Prediksi kelas
    with torch.no_grad():
        logits = model(input_tensor.to(next(model.parameters()).device))
        pred = logits.argmax(dim=1).item()

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    else:
        axes = ax

    axes[0].imshow(original)
    axes[0].set_title(f"Original\nLabel: {class_names[label]}", fontsize=10)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlaid)
    color = "green" if pred == label else "red"
    axes[2].set_title(f"Overlay\nPredicted: {class_names[pred]}", fontsize=10, color=color)
    axes[2].axis("off")

    if ax is None:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    return heatmap


def batch_gradcam(model, target_layer, data_loader, n_samples=8,
                  class_names=None, save_path=None):
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    device = next(model.parameters()).device
    model.eval()

    images, labels = [], []
    for batch in data_loader:
        batch_imgs, batch_labels, *_ = batch  # toleran terhadap dataloader yang return ekstra nilai
        for img, lbl in zip(batch_imgs, batch_labels):
            images.append(img)
            labels.append(lbl.item())
            if len(images) >= n_samples:
                break
        if len(images) >= n_samples:
            break

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i, (img, lbl) in enumerate(zip(images, labels)):
        inp = img.unsqueeze(0)
        visualize_gradcam(
            model, target_layer, inp, lbl,
            class_names=class_names,
            ax=axes[i]
        )

    plt.suptitle("Grad-CAM Analysis — Chest X-Ray", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Batch Grad-CAM saved: {save_path}")
    plt.show()