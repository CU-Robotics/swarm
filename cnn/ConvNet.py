import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    
    # defining the different layers of the CNN
    def __init__(self):
        super(ConvNet, self).__init__()
        # Input: [1, 100, 100]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # → [32, 100, 100]
        self.pool = nn.MaxPool2d(2, 2)                           # → [32, 50, 50]
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # → [64, 50, 50]
        # pool again
        # → [64, 25, 25]
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # → [128, 25, 25]
        # pool again
        # → [128, 12, 12]

        # Flatten: 128*12*12 = 18432
        self.fc1 = nn.Linear(128*12*12, 256)
        self.fc2 = nn.Linear(256, 10)  # for 10 classes (example)

    # how we want to pass the data through the network
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # image -> conv1 -> relu -> pool
        x = self.pool(F.relu(self.conv2(x))) # -> conv2 -> relu -> pool
        x = self.pool(F.relu(self.conv3(x))) # -> conv3 -> relu -> pool
        x = x.view(x.size(0), -1)  # flatten 
        x = F.relu(self.fc1(x)) # fully connected layer 1 -> relu
        x = self.fc2(x) # fully connected layer 2
        return x