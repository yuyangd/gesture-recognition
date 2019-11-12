import torchvision
import torch
from PIL import Image
from pathlib import Path
from torchvision.transforms import transforms, ToTensor
import torch.nn.functional as F

model = torchvision.models.resnet18()

model.fc = torch.nn.Linear(512, 2)

model.load_state_dict(torch.load('duy_model.pth', map_location=torch.device('cpu')))



# read image

image = Image.open(Path('pred_img/eb0993b6-8f2d-11e9-8dd0-e2e803ec7935.jpg'))

# transform to tensor
image = ToTensor()(image).unsqueeze(0) # unsqueeze to add artificial first dimension

output = model(image)

output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()

category_index = output.argmax()

print(output)

