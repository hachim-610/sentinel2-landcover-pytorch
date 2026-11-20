"""Exploration du jeu de données EuroSAT (Sentinel-2, 13 bandes)."""

from torchgeo.datasets import EuroSAT

# 1. Charger la partie "train" du jeu de données.
#    La première fois, download=True télécharge environ 2 Go dans data/eurosat
#    (dossier ignoré par Git grâce au .gitignore).
train_ds = EuroSAT(root="data/eurosat", split="train", download=True)

# 2. Taille du jeu de données
print("Nombre d'images (train) :", len(train_ds))
print("Classes :", train_ds.classes)

# 3. Récupérer le premier échantillon.
#    Chaque échantillon est un dictionnaire Python (dict) qui contient l'image et son label.
sample = train_ds[0]
print("Clés de l'échantillon :", list(sample.keys()))

image = sample["image"]
label = sample["label"]

# ---------------------------------------------------------------
# À toi de jouer : complète les TODO ci-dessous avec des print().
# ---------------------------------------------------------------

# TODO 1 : afficher la forme du tenseur image
print("forme :", image.shape)
# TODO 2 : afficher son type de données
print("type de données :", image.dtype)
# TODO 3 : afficher sa valeur minimale et sa valeur maximale
print("valeur min et max :", image.min(),  image.max())
# TODO 4 : afficher le label (un entier), puis le nom de la classe correspondante
print("label :", label.item(), "| Classe :",train_ds.classes[label.item()])
# TODO 5 : afficher la signature spectrale du pixel central (ligne 32, colonne 32),
#          c'est-à-dire ses 13 valeurs de 
print("signature spectrale du pixel 32x32 et valeurs des 13 bandes", image[:, 32, 32])