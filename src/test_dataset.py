import os
import torch
from dataset import LandCoverDataset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

txt_file = os.path.join(BASE_DIR, "LandCoverAI", "train.txt")

dataset = LandCoverDataset(txt_file=txt_file)

img, mask = dataset[0]

print("Image shape:", img.shape)   
print("Mask shape:", mask.shape)   
print("Unique values:", torch.unique(mask))