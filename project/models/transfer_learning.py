import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50TransferLearning(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50TransferLearning, self).__init__()
        
        # Load pretrained ResNet50 dari ImageNet
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze seluruh parameter backbone bawaan
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Ganti classifier head untuk 2 kelas (NORMAL vs PNEUMONIA)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def unfreeze_layer4(self):
        """Membuka layer4 agar bisa di-fine-tune"""
        for param in self.model.layer4.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)