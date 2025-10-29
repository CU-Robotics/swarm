import torch
import os
import torchvision.transforms as transforms
from PIL import Image

# load model
model = torch.load(os.getcwd() + '/models/full_model.pth', map_location=torch.device('cpu'), weights_only=False)

# 100x100 fake image
fake_image = torch.randn(1, 100, 100)  # 1 channel

# convert fake_image to PIL image
fake_image = transforms.ToPILImage()(fake_image)

transform = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.ToTensor(),
      transforms.Lambda(lambda t: t.sqrt()),
])

fake_image = transform(fake_image)
# add batch dimension
fake_image = fake_image.unsqueeze(0)

classes = {1:'1', 2:'2', 3:'3', 4:'4', 5:'sentry', 6:'base', 7:'tower'}
output = model(fake_image)
predicted_class = classes[output.argmax(dim=1).item()]

print(predicted_class)
