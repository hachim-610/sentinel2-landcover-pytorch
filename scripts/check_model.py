"""Vérifie que le modèle accepte un lot EuroSAT et produit une sortie de la bonne forme."""

import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT

from s2lc.data import Normalize, load_stats
from s2lc.model import build_model

# Utiliser le GPU s'il est disponible, sinon le CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# Un lot de données normalisées
mean, std = load_stats("configs/eurosat_stats.json")
train_ds = EuroSAT(root="data/eurosat", split="train", transforms=Normalize(mean, std))
loader = DataLoader(train_ds, batch_size=64, shuffle=True)
batch = next(iter(loader))

# Le modèle
model = build_model(in_channels=13, num_classes=10)
n_params = sum(p.numel() for p in model.parameters())
print(f"Nombre de paramètres : {n_params:,}")

# TODO 3 : envoyer le modèle sur le device
model = model.to(device)

# TODO 4 : envoyer les images du lot sur le device
images = batch["image"].to(device)
labels = batch["label"]

# Passage avant (forward) : les images traversent le réseau
logits = model(images)

print("Forme des images :", images.shape, "| sur :", images.device)
print("Forme des logits :", logits.shape)
print("Logits de la 1re image :", logits[0])
print("Classe prédite pour les 8 premières images :", logits.argmax(dim=1)[:8])
print("Vraies classes des 8 premières images      :", labels[:8])