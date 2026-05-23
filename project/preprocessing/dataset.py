import torch
from torch.utils.data import Dataset
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'path']
        label = self.dataframe.loc[idx, 'label']

        # Load image menggunakan PIL
        image = Image.open(img_path).convert('RGB') # Handling grayscale/RGB

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)