"""Diagnostic de l'entraînement : valeurs extrêmes et sur-apprentissage d'un seul lot."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT

from s2lc.data import Normalize, load_stats
from s2lc.model import build_model

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean, std = load_stats("configs/eurosat_stats.json")
train_ds = EuroSAT(root="data/eurosat", split="train", transforms=Normalize(mean, std))
loader = DataLoader(train_ds, batch_size=64, shuffle=True)
batch = next(iter(loader))
images = batch["image"].to(device)
labels = batch["label"].to(device)

# --- Test 1 : valeurs extrêmes dans les données normalisées ---
print("=== Test 1 : valeurs des données normalisées ===")
print(f"Minimum : {images.min().item():.2f} | Maximum : {images.max().item():.2f}")
print("Présence de NaN :", torch.isnan(images).any().item())
print("Présence d'infini :", torch.isinf(images).any().item())
print("Valeur absolue maximale par bande :")
print(images.abs().amax(dim=(0, 2, 3)))

# --- Test 2 : sur-apprentissage d'un seul lot, avec deux taux d'apprentissage ---
for lr in [1e-3, 1e-4]:
    print(f"\n=== Test 2 : un seul lot, lr = {lr} ===")
    torch.manual_seed(42)
    model = build_model(in_channels=13, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for step in range(100):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 99:
            acc = (logits.argmax(dim=1) == labels).float().mean().item()
            print(f"  Pas {step:3d} | perte {loss.item():.4f} | précision {acc:.2f}")