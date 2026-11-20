"""Construction du modèle : ResNet-18 adapté aux images Sentinel-2 multispectrales."""

import torch.nn as nn
from torchvision.models import resnet18


def build_model(in_channels=13, num_classes=10):
    """Crée un ResNet-18 adapté au nombre de bandes en entrée et au nombre de classes.

    Le modèle part de poids aléatoires (pas de pré-entraînement ImageNet).
    """
    model = resnet18(weights=None)

    # TODO 1 : remplacer la première convolution (prévue pour 3 canaux RGB)
    #          par une convolution identique, mais qui accepte in_channels canaux
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # TODO 2 : remplacer la dernière couche (1000 classes ImageNet)
    #          par une couche qui produit num_classes valeurs
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model