import torch
from torch.utils.data import DataLoader
import os

from dataset import LandCoverDataset
from model import UNet
from losses import PFCE_Loss


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.path.join(BASE_DIR, "LandCoverAI")

train_dataset = LandCoverDataset(
    txt_file=os.path.join(BASE_PATH, "train.txt"),
    mode="points",
    num_points=50
)

val_dataset = LandCoverDataset(
    txt_file=os.path.join(BASE_PATH, "val.txt"),
    mode="points",
    num_points=50
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(3, 5).to(device)

criterion = PFCE_Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            val_loss += loss.item()

    print(f"Epoch {epoch+1} Val Loss: {val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), "model.pth")
print("Saved ")