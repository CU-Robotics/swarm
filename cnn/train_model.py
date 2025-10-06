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

    # Split sizes
    train_size = int(0.8 * len(dataset))  # 80%
    val_size = len(dataset) - train_size  # remaining 20%

    # Random split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    

    net = ConvNet.ConvNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1:.3f}')
            running_loss = 0.0
    
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    print('Finished Training')

    # Test the network on the validation dataset while also showing image and guessing label
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in val_loader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #         for j, image in enumerate(images):
    #             cv2.imshow('Image', (image.numpy().transpose((1, 2, 0)) * 255).astype('uint8'))
    #             print('Predicted: ', ' '.join(f'{predicted[j]}'))
    #             print('Actual: ', ' '.join(f'{labels[j]}'))
    #             cv2.waitKey(0)  # Wait for a key press to move to the next image
    
    # cv2.destroyAllWindows()

    # show images of kernel applied to images
    import torch.nn.functional as F
    import cv2

    correct = 0
    total = 0

    import cv2
    import torch.nn.functional as F
    import numpy as np

    def to_img(tensor):
        """Normalize tensor to 0-255 uint8 image."""
        tensor = tensor.clone()
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)
        return (tensor * 255).byte().cpu().numpy()

    def to_bgr(img):
        """Ensure image is 3-channel BGR for cv2."""
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def resize_to_height(img, target_height):
        """Resize an image while keeping aspect ratio to a target height."""
        return cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height))

    total = 0
    correct = 0
    debug = False

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if debug:
                for idx in range(len(images)):
                    image = images[idx].unsqueeze(0)  # [1,3,H,W]
                    label = labels[idx]

                    # ---- original ----
                    orig = to_img(image.squeeze(0)).transpose(1,2,0)  # HWC

                    # ---- conv1 ----
                    x1 = net.conv1(image)
                    x1_img = to_img(x1[0,0])

                    # ---- pool1 ----
                    x1_pool = net.pool(x1)
                    x1_pool_img = to_img(x1_pool[0,0])

                    # ---- conv2 ----
                    x2 = net.conv2(x1_pool)
                    x2_img = to_img(x2[0,0])

                    # ---- pool2 ----
                    x2_pool = net.pool(x2)
                    x2_pool_img = to_img(x2_pool[0,0])

                    # Resize all feature maps to original height
                    target_height = orig.shape[0]
                    x1_img_resized = resize_to_height(x1_img, target_height)
                    x1_pool_resized = resize_to_height(x1_pool_img, target_height)
                    x2_img_resized = resize_to_height(x2_img, target_height)
                    x2_pool_resized = resize_to_height(x2_pool_img, target_height)

                    combined = cv2.hconcat([
                        orig.astype('uint8'),
                        to_bgr(x1_img_resized),
                        to_bgr(x1_pool_resized),
                        to_bgr(x2_img_resized),
                        to_bgr(x2_pool_resized)
                    ])
                    
                    cv2.imshow('Feature Maps', combined)
                    print(f'Label: {label.item()}')
                    print(f'Predicted: {predicted[idx].item()}')
                    cv2.waitKey(0)

                cv2.destroyAllWindows()


    print(f'Accuracy of the network on the {total} validation images: {100 * correct / total} %')




 
if __name__ == "__main__":
    main()
    # add return to new folder name
    #/collections/armor_plates_9-14-25/examples/raw