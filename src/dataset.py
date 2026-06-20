import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class LandCoverDataset(Dataset):

    def __init__(self, txt_file, mode="full", num_points=50):
        with open(txt_file, "r") as f:
            self.data = f.readlines()

        self.mode = mode
        self.num_points = num_points

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        img_path, mask_path = line.split()

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = self.transform(image)
        mask = mask.resize((512, 512))

        mask = torch.tensor(np.array(mask), dtype=torch.long)

        if self.mode == "points":
            mask = generate_point_mask(mask, self.num_points)

        return image, mask


def generate_point_mask(mask, num_points=50):
    point_mask = torch.full_like(mask, -1)

    classes = torch.unique(mask)

    for cls in classes:
        if cls == -1:
            continue

        coords = (mask == cls).nonzero(as_tuple=False)

        if len(coords) == 0:
            continue

        idx = torch.randperm(len(coords))[:num_points]
        selected = coords[idx]

        for y, x in selected:
            point_mask[y, x] = cls

    return point_mask