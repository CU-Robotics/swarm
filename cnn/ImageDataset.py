import os
import json
from torchvision.io import decode_image
import torch
from torch.utils.data import Dataset
from  PIL import Image

#Note: CHANGE ALL INSTANCES OF "examples"

classes = {'1':1, '2':2, '3':3, '4':4, 'sentry':5, 'base':6, 'tower':7}

# Custom dataset class for loading images and labels
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = json.load(open(annotations_file)) # path to pipeline.json
        self.img_dir = img_dir                              # folder where the cleaned images are
        self.transform = transform                          

    def __len__(self):
        return len(self.img_labels["examples"])

    # gets image and label at index idx based on position in json
    def __getitem__(self, idx):
        img_name = self.img_labels["examples"][idx]["name"]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        label = classes[self.img_labels["examples"][idx]["labels"]["icon"]] 

        if self.transform:
            image = self.transform(image)
        
        return image, label

