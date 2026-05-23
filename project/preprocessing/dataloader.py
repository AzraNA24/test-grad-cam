import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Import class
from preprocessing.dataset import ChestXrayDataset
from preprocessing.preprocessing import train_transforms, test_val_transforms

# LOAD PATHS DAN CEK CLASS DISTRIBUTION
dataset_dir = '/content/dataset/chest_xray'

# Mengumpulkan semua path gambar
normal_paths = glob.glob(os.path.join(dataset_dir, '**', 'NORMAL', '*.jpeg'), recursive=True)
pneumonia_paths = glob.glob(os.path.join(dataset_dir, '**', 'PNEUMONIA', '*.jpeg'), recursive=True)

# Membuat DataFrame
data = []
for path in normal_paths:
    data.append({'path': path, 'label': 0, 'class': 'NORMAL'})
for path in pneumonia_paths:
    data.append({'path': path, 'label': 1, 'class': 'PNEUMONIA'})

df = pd.DataFrame(data)

# Cek Class Distribution
print("--- Class Distribution ---")
print(df['class'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='class', palette='Set2')
plt.title('Distribusi Kelas Dataset Chest X-Ray')
plt.show()

# SPLIT DATASET
# Split pertama: Train (70%) dan Temp (30% untuk Val & Test)
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)

# Split kedua: Bagi Temp menjadi Val (15%) dan Test (15%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"Total Data Train: {len(train_df)}")
print(f"Total Data Val: {len(val_df)}")
print(f"Total Data Test: {len(test_df)}")

# Reset index
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Buat instansiasi Dataset
train_dataset = ChestXrayDataset(train_df, transform=train_transforms)
val_dataset = ChestXrayDataset(val_df, transform=test_val_transforms)
test_dataset = ChestXrayDataset(test_df, transform=test_val_transforms)

BATCH_SIZE = 32

# Integrasi dengan transform, batching, dan shuffle
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("DataLoader berhasil dibuat!")

# CEK DATALOADER (VISUALISASI)
def imshow(inp, title=None):
    """Fungsi untuk denormalize dan menampilkan tensor gambar"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Ambil satu batch dari train_loader
inputs, classes = next(iter(train_loader))

# Buat grid dari batch
import torchvision
out = torchvision.utils.make_grid(inputs[:4]) # Ambil 4 gambar saja untuk preview
class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}

plt.figure(figsize=(12, 4))
imshow(out, title=[class_names[x.item()] for x in classes[:4]])