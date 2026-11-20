"""Vérifie que la normalisation donne une moyenne ~0 et un écart-type ~1 par bande."""

from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT

from s2lc.data import Normalize, load_stats

mean, std = load_stats("configs/eurosat_stats.json")

train_ds = EuroSAT(
    root="data/eurosat",
    split="train",
    transforms=Normalize(mean, std),
)

# shuffle=True : on veut un lot représentatif de toutes les classes
loader = DataLoader(train_ds, batch_size=256, shuffle=True)

# Récupérer un seul lot
batch = next(iter(loader))
images = batch["image"]
print("Forme du lot :", images.shape)

# TODO 3 : afficher la moyenne de chaque bande sur ce lot (13 valeurs)
print("Moyennes par bande :", images.mean(dim=(0, 2, 3)))
# TODO 4 : afficher l'écart-type de chaque bande sur ce lot (13 valeurs)
print("Écart-type par bande :", images.std(dim=(0, 2, 3)))