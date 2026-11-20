"""Outils de chargement et de prétraitement des données EuroSAT."""

import json

import torch


def load_stats(path):
    """Lit les moyennes et écarts-types par bande depuis un fichier JSON."""
    with open(path) as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


class Normalize:
    """Standardise chaque bande d'une image : (x - moyenne) / écart-type."""

    def __init__(self, mean, std):
        # TODO 1 : convertir mean et std (des listes Python de 13 valeurs)
        #          en tenseurs float32 de forme (13, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, sample):
        # TODO 2 : normaliser l'image de l'échantillon avec self.mean et self.std
        sample["image"] = (sample["image"] - self.mean) / self.std
        return sample