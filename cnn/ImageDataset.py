import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

classes = {'1':1, '2':2, '3':3, '4':4, 'sentry':5, 'base':6, 'tower':7}

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = json.load(open(annotations_file)) # path to pipeline.json
        self.img_dir = img_dir # folder where the cleaned images are
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels["examples"]) # change when we decide on how the pipeline will be made

    def __getitem__(self, idx):
        img_name = self.img_labels["examples"][idx]["name"]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        label = classes[self.img_labels["examples"][idx]["labels"]["icon"]] # gets label from json file and converts to number based on "classes"

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label

