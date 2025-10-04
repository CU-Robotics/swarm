import ImageDataset
import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
def main():
    if len(sys.argv) == 2:
            print("Arguments passed:" + sys.argv[1])
    else:
        print("Incorrect number of arguments passed. Please ONLY provide the folder path to be processed.")
        exit()
        
    datadir = os.getcwd() + sys.argv[1]
    parent_dir = os.path.dirname(datadir)
    parent_folder_name = os.path.basename(parent_dir)

    print("Reading from directory: " + datadir + "\n")


    batch_size = 4

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = ImageDataset.CustomImageDataset(annotations_file=parent_dir + '/metadata.json', img_dir=datadir, transform=transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    
    

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    import ConvNet
    net = ConvNet.ConvNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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




    
if __name__ == "__main__":
    main()
    # add return to new folder name
    #/collections/armor_plates_9-14-25/examples/raw