import torch

import ImageDataset
import torch
import os
import sys
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

import ConvNet

def main():
    if len(sys.argv) == 2:
            print("Arguments passed:" + sys.argv[1])
    else:
        print("Incorrect number of arguments passed. Please ONLY provide the folder path to be processed.")
        exit()
    
    # seed for reproducibility
    torch.manual_seed(389)

    datadir = os.getcwd() + sys.argv[1]
    parent_dir = os.path.dirname(datadir)
    parent_folder_name = os.path.basename(parent_dir)

    print("Reading from directory: " + datadir + "\n")


    batch_size = 4

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageDataset.CustomImageDataset(annotations_file=parent_dir + '/combined_metadata.json', img_dir=datadir, transform=transform)

   

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    net = torch.load("cifar_net.pth")

    total = 0
    correct = 0


    with torch.no_grad():
        for images, labels in test_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()



    print(f'Accuracy of the network on the {total} validation images: {100 * correct / total} %')




 
if __name__ == "__main__":
    main()
    # add return to new folder name
    #/collections/armor_plates_9-14-25/examples/raw