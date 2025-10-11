import ImageDataset
import torch
import os
import sys
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
import ConvNet


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f'Using device: {device}')

def train(dataloader: DataLoader, model:nn.Module, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader: DataLoader, model:nn.Module, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (test_loss, correct)

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
    print("Reading from directory: " + datadir + "\n")


    batch_size = 4

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    annotations_file = os.path.join(parent_dir, 'combined_metadata.json')
    dataset = ImageDataset.CustomImageDataset(annotations_file, img_dir=datadir, transform=transform)

    # Split sizes
    train_size = int(0.8 * len(dataset))  # 80%
    val_size = len(dataset) - train_size  # remaining 20%

    # Random split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    net = ConvNet.ConvNet()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_loader, net, criterion, optimizer)  
    
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    print('Finished Training')

    test(val_loader, net, criterion)

    
if __name__ == "__main__":
    main()
    