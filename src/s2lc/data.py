"""Outils de chargement et de prétraitement des données EuroSAT."""

import json

import torch


def load_stats(path):
    """Lit les moyennes et écarts-types par bande depuis un fichier JSON."""
    with open(path) as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


class Normalize:
    """Standardise chaque bande d'une image : (x - moyenne) / écart-type.

    Si clip est donné, les valeurs normalisées sont écrêtées dans [-clip, clip],
    pour limiter l'effet des pixels extrêmes (nuages, reflets, surfaces très brillantes).
    """

    def __init__(self, mean, std, clip=None):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.clip = clip

    def __call__(self, sample):
        image = (sample["image"] - self.mean) / self.std

        if self.clip is not None:
            # TODO : écrêter image entre -self.clip et +self.clip
            image = image.clamp(-self.clip, self.clip)

        sample["image"] = image
        return sample