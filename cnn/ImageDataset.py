import os
import json
from torchvision.io import decode_image
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = json.load(open(annotations_file))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels["examples"])

    def __getitem__(self, idx):
        img_name = self.img_labels["examples"][idx]["name"]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = decode_image(img_path)
        label = self.img_labels["examples"][idx]["labels"]["icon"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

