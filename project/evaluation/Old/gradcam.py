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
        score = output[0, class_idx]
        score.backward()

        # Compute heatmap
        # gradients  : (1, C, H, W)
        # activations: (1, C, H, W)
        gradients   = self.gradients[0]   
        activations = self.activations[0]        

        # Global Average Pooling pada gradients 
        weights = gradients.mean(dim=(1, 2))      

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU ambil hanya kontribusi positif
        cam = F.relu(cam)

        # Resize ke ukuran input asli
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = cam.unsqueeze(0).unsqueeze(0)        # (1, 1, h, w)
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()          # (H, W)

        # Normalize 0–1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, pred_prob

    # ------------------------------------------------------------------
    # Overlay heatmap ke gambar asli
    # ------------------------------------------------------------------

    @staticmethod
    def overlay(image_np, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        # Pastikan image_np adalah uint8 BGR
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)

        if len(image_np.shape) == 2:                         # grayscale → BGR
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] == 1:
            image_np = cv2.cvtColor(image_np[:, :, 0], cv2.COLOR_GRAY2BGR)

        # Konversi heatmap 0-1 → 0-255 → colormap BGR
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

        # Overlay
        overlay_img = cv2.addWeighted(heatmap_colored, alpha, image_np, 1 - alpha, 0)

        return overlay_img, heatmap_colored

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

def get_target_layer(model):
    model_name = type(model).__name__.lower()

    # ResNet variants
    if "resnet" in model_name:
        return model.layer4[-1]

    # EfficientNet (timm / torchvision)
    if "efficientnet" in model_name:
        try:
            return model.features[-1]
        except AttributeError:
            return list(model.children())[-3]

    # BASELINE
    for attr in ["conv_layers", "features", "conv3", "conv_block3"]:
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, torch.nn.Sequential):
                return layer[-1]
            return layer

    # Fallback: cari module konvolusi terakhir secara generik
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        print("[GradCAM] Menggunakan Conv2d terakhir yang ditemukan sebagai target layer.")
        return last_conv

    raise ValueError(
        "Tidak bisa auto-detect target layer. "
        "Panggil GradCAM(model, target_layer=...) secara manual."
    )
