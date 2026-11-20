"""Signatures spectrales moyennes par classe (EuroSAT, jeu d'entraînement)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT

# Données brutes (sans normalisation) : on veut des réflectances
train_ds = EuroSAT(root="data/eurosat", split="train")
loader = DataLoader(train_ds, batch_size=256, shuffle=False)

n_classes = len(train_ds.classes)  # 10
n_bands = 13

# "Tirelires" : une ligne par classe, une colonne par bande
class_sum = torch.zeros(n_classes, n_bands, dtype=torch.float64)  # forme (10, 13)
class_n_pixels = torch.zeros(n_classes, dtype=torch.float64)       # forme (10,)

# --- 1. Accumulation, lot par lot et classe par classe ---
for batch in loader:
    images = batch["image"].double()  # forme (B, 13, 64, 64)
    labels = batch["label"]           # forme (B,)
    pixels_per_image = images.shape[2] * images.shape[3]  # 64 x 64

    for c in range(n_classes):
        # TODO 1 : masque booléen indiquant quelles images du lot sont de la classe c
        mask = (labels == c)

        # TODO 2 : ajouter la somme par bande des images de la classe c (forme (13,))
        class_sum[c] += images[mask].sum(dim=(0, 2, 3))

        # TODO 3 : ajouter le nombre de pixels comptés pour la classe c
        class_n_pixels[c] += mask.sum() * pixels_per_image

# --- 2. Moyennes ---
# TODO 4 : moyenne par classe et par bande (forme (10, 13)), convertie en réflectance
mean_refl = class_sum / (class_n_pixels.view(-1, 1) * 10000)

# --- 3. Tri des bandes par longueur d'onde ---
# Longueurs d'onde centrales (nm), dans l'ordre des bandes de TorchGeo (B8A en dernier)
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B09", "B10", "B11", "B12", "B8A"]
WAVELENGTHS = torch.tensor([443, 490, 560, 665, 705, 740, 783,
                            842, 945, 1375, 1610, 2190, 865])

# TODO 5 : obtenir l'ordre des bandes trié par longueur d'onde, puis réordonner
order = torch.argsort(WAVELENGTHS)
wl_sorted = WAVELENGTHS[order]
mean_refl_sorted = mean_refl[:, order]

# --- 4. Figure ---
fig, ax = plt.subplots(figsize=(11, 6))
for c, name in enumerate(train_ds.classes):
    ax.plot(wl_sorted, mean_refl_sorted[c], marker="o", label=name)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Mean TOA reflectance")
ax.set_title("EuroSAT – mean spectral signature per class (train split)")
ax.grid(alpha=0.3)
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()

out_path = Path("figures/eurosat_spectral_signatures.png")
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=150)
print("Figure enregistrée dans", out_path)

plt.show()