"""Affiche une imagette EuroSAT par classe, en couleurs naturelles (B04, B03, B02)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchgeo.datasets import EuroSAT

# Données brutes, SANS normalisation : on veut afficher des réflectances
train_ds = EuroSAT(root="data/eurosat", split="train")

RGB_IDX = [3, 2, 1]  # positions de B04, B03, B02 dans l'ordre des bandes de TorchGeo
# RGB_IDX = [7, 3, 2] # représentation en fausse couleur IR (B08, B04, B03)
STRETCH_MAX = 3000 # valeur L1C affiché à blanc (réflectance 0,3)

# --- 1. Trouver une image au hasard pour chaque classe ---
torch.manual_seed(0)  # graine fixée : le même tirage à chaque exécution
n_classes = len(train_ds.classes)
chosen = {}  # dictionnaire {numéro de classe : indice de l'image choisie}

for idx in torch.randperm(len(train_ds)).tolist():
    label = train_ds[idx]["label"].item()
    if label not in chosen:
        chosen[label] = idx
    if len(chosen) == n_classes:
        break  # une image trouvée pour chaque classe : on arrête

# --- 2. Construire la figure : 2 lignes x 5 colonnes ---
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for label, ax in zip(sorted(chosen), axes.flat):
    image = train_ds[chosen[label]]["image"]  # forme (13, 64, 64)

    # TODO 1 : extraire les 3 bandes RGB, dans l'ordre B04, B03, B02 -> forme (3, 64, 64)
    rgb = image[RGB_IDX]

    # TODO 2 : réordonner les dimensions en (64, 64, 3), l'ordre attendu par matplotlib
    rgb = rgb.permute(1, 2, 0)

    # TODO 3 : étirement : valeurs 0–3000 (réflectance 0–0,3) ramenées entre 0 et 1,
    #          les valeurs au-delà étant plafonnées à 1
    rgb = (rgb/STRETCH_MAX).clamp(0, 1)

    ax.imshow(rgb.numpy())
    ax.set_title(train_ds.classes[label])
    ax.axis("off")

fig.suptitle("EuroSAT – one sample per class (true colour, B4-B3-B2)")
fig.tight_layout()

# --- 3. Enregistrer la figure ---
out_path = Path("figures/eurosat_samples.png")
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=150)
print("Figure enregistrée dans", out_path)

plt.show()