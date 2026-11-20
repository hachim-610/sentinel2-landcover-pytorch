"""Calcul de la moyenne et de l'écart-type de chaque bande
sur le jeu d'entraînement EuroSAT (13 bandes Sentinel-2)."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT

# Jeu d'entraînement uniquement (pour éviter toute fuite de données)
train_ds = EuroSAT(root="data/eurosat", split="train", download=False)

# DataLoader : lots de 256 images.
# shuffle=False : l'ordre ne change rien à une moyenne, inutile de mélanger ici.
train_loader = DataLoader(train_ds, batch_size=256, shuffle=False)

n_bands = 13
band_sum = torch.zeros(n_bands, dtype=torch.float64)     # somme par bande
band_sq_sum = torch.zeros(n_bands, dtype=torch.float64)  # somme des carrés par bande
n_pixels = 0                                             # pixels comptés par bande

for i, batch in enumerate(train_loader):
    images = batch["image"].double()  # conversion en float64

    if i == 0:
        # TODO 1 : afficher la forme de images, puis celle de batch["label"]
        print("forme des images :", images.shape, "| forme des labels :", batch["label"].shape)

    # TODO 2 : ajouter à band_sum la somme de chaque bande sur ce lot (forme (13,))
    band_sum += images.sum(dim=(0, 2, 3))

    # TODO 3 : même chose pour les valeurs au carré
    band_sq_sum += (images**2).sum(dim=(0, 2, 3))

    # TODO 4 : ajouter le nombre de pixels par bande de ce lot (B × H × W)
    n_pixels += images.shape[0]*images.shape[2]*images.shape[3]

    if i % 10 == 0:
        print(f"Lot {i + 1}/{len(train_loader)}")

mean = band_sum / n_pixels
std = torch.sqrt(band_sq_sum / n_pixels - mean**2)

print("Moyennes     :", mean)
print("Écarts-types :", std)

# Sauvegarde dans un fichier JSON (format texte lisible, versionné avec Git)
out_path = Path("configs/eurosat_stats.json")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
print("Statistiques enregistrées dans", out_path)