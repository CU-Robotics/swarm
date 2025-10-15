import ImageDataset
import torch
import os
import sys
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
import ConvNet
from tqdm import tqdm


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f'Using device: {device}')

def train(dataloader: DataLoader, model:nn.Module, loss_fn, optimizer):
    model.train()
    running_loss = 0.0

    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=True)
    for batch, (X, y) in loop:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def test(dataloader: DataLoader, model:nn.Module, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        loop = tqdm(dataloader, total=len(dataloader), desc="Validation", leave=False)
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()                
    
    test_loss /= num_batches
    correct /= size

    return (test_loss, correct)

def multi_train(train_dataloader: DataLoader, val_dataloader: DataLoader, model:nn.Module, loss_fn, optimizer, epochs: int):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        train_loss, train_acc = test(train_dataloader, model, loss_fn)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"\nTraining Loss: {round(train_loss, 5)}\nTraining Accuracy: {round(train_acc, 5)}\n")

        val_loss, val_acc = test(val_dataloader, model, loss_fn)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Validation Loss: {round(val_loss, 5)}\nValidation Accuracy: {round(val_acc, 5)}\n")
    

    return train_losses, val_losses, train_accuracies, val_accuracies

def main():
    if len(sys.argv) == 2:
            print("Arguments passed:" + sys.argv[1])
    else:
        print("Incorrect number of arguments passed. Please ONLY provide the folder path to be processed.")
        exit()
    
    # seed for reproducibility
    torch.manual_seed(389)
    # print(os.getcwd())
    datadir = os.getcwd() + sys.argv[1]
    # print("Datadir: " + datadir + "\n")
    parent_dir = os.path.dirname(datadir)
    # print("Parent dir: " + parent_dir + "\n")
    # print("Reading from directory: " + datadir + "\n")

    batch_size = 16

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(0.5, 0.5)
    ])

    annotations_file = os.path.join(datadir, 'cleaned_metadata.json')
    # print("Using parentdir file: " + parent_dir + "\n")
    
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

    multi_train(train_loader, val_loader, net, criterion, optimizer, epochs=10) 
    
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    print('Finished Training')

    test(val_loader, net, criterion)

    
if __name__ == "__main__":
    main()
    