import os
import torch
from torch.utils.data import DataLoader

from dataset import LandCoverDataset
from model import UNet
from losses import PFCE_Loss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.path.join(BASE_DIR, "LandCoverAI")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Train function
def train_model(num_points=50, epochs=3):

    train_dataset = LandCoverDataset(
        txt_file=os.path.join(BASE_PATH, "train.txt"),
        mode="points",
        num_points=num_points
    )

    val_dataset = LandCoverDataset(
        txt_file=os.path.join(BASE_PATH, "val.txt"),
        mode="points",
        num_points=num_points
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    model = UNet(3, 5).to(device)

    criterion = PFCE_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"[Points={num_points}] Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")

    # validation loss
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    return val_loss



# RUN Experiments

if __name__ == "__main__":

    experiments = [10, 50, 100]

    results = {}

    for p in experiments:
        print("\n========================")
        print(f"Running experiment: {p} points")
        print("========================")

        val_loss = train_model(num_points=p, epochs=3)

        results[p] = val_loss

 
    # PRINT final results
   
    print("\n\n===== FINAL EXPERIMENT RESULTS =====")

    for k, v in results.items():
        print(f"Points: {k} -> Val Loss: {v:.4f}")