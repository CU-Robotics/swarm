# Using the model trained in train_model.py, test its performance on a test dataset.
import os
import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import ConvNet
import ImageDataset
import train_model
import matplotlib.pyplot as plt


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f'Using device: {device}')

def main():
    if len(sys.argv) == 2:
        print("Arguments passed:" + sys.argv[1])
    else:
        print("Incorrect number of arguments passed. Please ONLY provide the folder path to be processed.")
        exit()
        
    datadir = os.getcwd() + "/" +  sys.argv[1]
    parent_dir = os.path.dirname(datadir)

    # Load the trained model
    model_path = os.path.join(parent_dir, 'cifar_net.pth')
    if not os.path.exists(model_path):
        print(f"Error: {model_path} does not exist. Please train the model first.")
        exit()
    
    model = ConvNet.ConvNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Define transformations (should match those used during training)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    annotations_file = os.path.join(datadir, 'cleaned_metadata.json')
    # print("Using parentdir file: " + parent_dir + "\n")
    
    dataset = ImageDataset.CustomImageDataset(annotations_file, img_dir=datadir, transform=transform)
    
    # Split dataset into training and testing sets (80% train, 20% test)
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()

    test_loss, test_acc = train_model.test(test_dataloader, model, loss_fn)
    print(f"Test Set - Loss: {round(test_loss, 5)}, Accuracy: {test_acc}")
    
    # Optional: Visualize some predictions
    classes = ["1", "2", "3", "4", "sentry", "base", "tower"]

    incorrect_images = []

    
    dataiter = iter(test_dataloader)

    for data in dataiter:
        images, labels = data
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            if predicted[i] != labels[i].to(device):
                incorrect_images.append((images[i], predicted[i], labels[i], outputs[i]))
    
    for idx in range(min(8, len(incorrect_images))):
        img, pred, true, outputs = incorrect_images[idx]
        
        #normalize outputs vector to sum to 1
        normalized_outputs = torch.softmax(outputs, dim=0)
        
        

        img = img.numpy().squeeze()
        img = (img * 0.5) + 0.5  # unnormalize
        plt.subplot(2, 4, idx+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {classes[pred.item() - 1]}({round(normalized_outputs[pred.item()].item(), 3)})\nTrue: {classes[true.item() - 1]}({round(normalized_outputs[true.item()].item(), 3)})", color="red")
        plt.axis('off')
    plt.show()
    
    # images, labels = next(dataiter)
    # images, labels = images.to(device), labels.to(device)

    # outputs = model(images)
    # _, predicted = torch.max(outputs, 1)

    # # Move tensors to CPU for visualization
    # images = images.cpu()
    # labels = labels.cpu()
    # predicted = predicted.cpu()

    # fig = plt.figure(figsize=(12, 6))
    # for idx in range(min(8, len(images))):
    #     ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
    #     img = images[idx].numpy().squeeze()
    #     img = (img * 0.5) + 0.5  # unnormalize
    #     plt.imshow(img, cmap='gray')

    #     pred_label = classes[predicted[idx].item() - 1]
    #     true_label = classes[labels[idx].item() - 1]

    #     ax.set_title(
    #         f"Pred: {pred_label}\nTrue: {true_label}",
    #         color=("green" if predicted[idx] == labels[idx] else "red")
    #     )
    # plt.show()


if __name__ == "__main__":
    main()
