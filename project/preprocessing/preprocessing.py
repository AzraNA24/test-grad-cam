import torchvision.transforms as transforms

# Mean dan STD ImageNet untuk normalisasi (standar jika menggunakan pre-trained CNN)
# Jika tidak pakai pre-trained, bisa gunakan mean=[0.5], std=[0.5]
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

# Augmentasi untuk Data Train
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize image
    transforms.Grayscale(num_output_channels=3), # Grayscale to RGB handling
    transforms.RandomHorizontalFlip(p=0.5),      # Flip
    transforms.RandomRotation(degrees=10),       # Rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Brightness/Contrast
    transforms.ToTensor(),
    normalize                                    # Normalization
])

# Preprocessing untuk Data Validasi/Test
test_val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    normalize
])