"""Extraction de caractéristiques simples pour les modèles classiques (Random Forest)."""

import torch
from torch.utils.data import DataLoader


def extract_features(dataset, batch_size=256):
    """Résume chaque image par la moyenne et l'écart-type de chacune de ses bandes.

    Renvoie :
        X : tableau numpy de forme (N, 2 * n_bandes), une ligne par image
        y : tableau numpy de forme (N,), le label de chaque image
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []  # liste des caractéristiques de chaque lot
    all_labels = []    # liste des labels de chaque lot

    for batch in loader:
        images = batch["image"]  # forme (B, C, H, W)

        # TODO 1 : moyenne de chaque bande, pour chaque image -> forme (B, C)
        means = images.mean(dim=(2, 3))

        # TODO 2 : écart-type de chaque bande, pour chaque image -> forme (B, C)
        stds = images.std(dim=(2, 3))

        # TODO 3 : mettre côte à côte les moyennes et les écarts-types -> forme (B, 2C)
        features = torch.cat([means, stds], dim=1)

        all_features.append(features)
        all_labels.append(batch["label"])

    # Rassembler tous les lots en un seul tableau, puis convertir en numpy
    X = torch.cat(all_features).numpy()
    y = torch.cat(all_labels).numpy()
    return X, y