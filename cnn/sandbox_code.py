import torch
import os
import torchvision.transforms as transforms


# load model

model = torch.load(os.getcwd() + '/models/full_model.pth', map_location=torch.device('cpu'), weights_only=False)

# 100, 100 fake image
fake_image = torch.randn(1, 1, 100, 100)  # Assuming the model expects a single-channel 100x100 image
classes = {1:'1', 2:'2', 3:'3', 4:'4', 5:'sentry', 6:'base', 7:'tower'}

transform = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.ToTensor(),
      transforms.Lambda(lambda t: t.sqrt()),
    ])

fake_image = transform(fake_image)
print(classes[model(fake_image).argmax(dim=1, keepdim=True).item()])

torch.tensor()