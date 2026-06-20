import torch
import matplotlib.pyplot as plt
import os

from dataset import LandCoverDataset
from model import UNet


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
txt_file = os.path.join(BASE_DIR, "LandCoverAI", "test.txt")

dataset = LandCoverDataset(txt_file, mode="full")

model = UNet(3, 5)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

img, mask = dataset[0]

with torch.no_grad():
    pred = model(img.unsqueeze(0))
    pred = torch.argmax(pred, dim=1).squeeze(0)

plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.title("Image")
plt.imshow(img.permute(1,2,0))

plt.subplot(1,3,2)
plt.title("GT")
plt.imshow(mask)

plt.subplot(1,3,3)
plt.title("Pred")
plt.imshow(pred)

plt.show()