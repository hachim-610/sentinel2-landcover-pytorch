"""Entraînement d'un ResNet-18 sur EuroSAT (13 bandes Sentinel-2)."""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT

from s2lc.data import Normalize, load_stats
from s2lc.model import build_model

# --- Réglages ---
N_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# --- Données ---
mean, std = load_stats("configs/eurosat_stats.json")
normalize = Normalize(mean, std, clip=5)
train_ds = EuroSAT(root="data/eurosat", split="train", transforms=normalize, download=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# --- Modèle, fonction de perte, optimiseur ---
model = build_model(in_channels=13, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Boucle d'entraînement ---
for epoch in range(N_EPOCHS):
    model.train()  # mode entraînement
    start = time.time()
    running_loss = 0.0
    n_correct = 0
    n_seen = 0

    for i, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # TODO 5 : remettre les gradients à zéro (on l'expliquera en dernier)
        optimizer.zero_grad()
        # TODO 1 : forward, faire passer les images dans le modèle
        logits = model(images)

        # TODO 2 : calculer la perte entre les prédictions et les vrais labels
        loss = criterion(logits, labels)

        # TODO 3 : backward, calculer les gradients
        loss.backward()
        # TODO 4 : faire un pas d'optimisation (mettre à jour les poids)
        optimizer.step()
        # --- Suivi de la progression ---
        running_loss += loss.item() * images.shape[0]
        n_correct += (logits.argmax(dim=1) == labels).sum().item()
        n_seen += images.shape[0]

        if (i + 1) % 50 == 0:
            print(f"  Époque {epoch + 1} | lot {i + 1}/{len(train_loader)} "
                  f"| perte {running_loss / n_seen:.4f} | précision {n_correct / n_seen:.3f}")

    print(f"Époque {epoch + 1}/{N_EPOCHS} terminée en {time.time() - start:.0f} s "
          f"| perte {running_loss / n_seen:.4f} | précision {n_correct / n_seen:.3f}")