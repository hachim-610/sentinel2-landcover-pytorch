"""Baseline Random Forest sur des caractéristiques spectrales simples (EuroSAT)."""

import time
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
)
from torchgeo.datasets import EuroSAT

from s2lc.features import extract_features

# Données brutes : un Random Forest n'a pas besoin de normalisation
train_ds = EuroSAT(root="data/eurosat", split="train")
val_ds = EuroSAT(root="data/eurosat", split="val", download=True)

# --- 1. Extraction des caractéristiques ---
print("Extraction des caractéristiques...")
X_train, y_train = extract_features(train_ds)
X_val, y_val = extract_features(val_ds)
print("X_train :", X_train.shape, "| X_val :", X_val.shape)

# --- 2. Entraînement ---
# TODO 4 : créer un Random Forest de 200 arbres, avec une graine fixée (42),
#          utilisant tous les cœurs du processeur
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

start = time.time()
# TODO 5 : entraîner le modèle sur les données d'entraînement
rf.fit(X_train, y_train)
print(f"Entraînement terminé en {time.time() - start:.1f} s")

# --- 3. Prédiction et évaluation sur la validation ---
# TODO 6 : prédire les classes des images de validation
y_pred = rf.predict(X_val)

print(f"Précision globale (OA) : {accuracy_score(y_val, y_pred):.4f}")
print(f"Kappa de Cohen         : {cohen_kappa_score(y_val, y_pred):.4f}")
print(classification_report(y_val, y_pred, target_names=train_ds.classes, digits=3))

# --- 4. Matrice de confusion ---
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay.from_predictions(
    y_val, y_pred,
    display_labels=train_ds.classes,
    normalize="true",
    values_format=".2f",
    xticks_rotation=45,
    cmap="Blues",
    ax=ax,
)
ax.set_title("Random Forest – confusion matrix (val split, row-normalized)")
fig.tight_layout()

out_path = Path("figures/rf_confusion_matrix.png")
fig.savefig(out_path, dpi=150)
print("Figure enregistrée dans", out_path)

#plt.show() # si affichage de la matrice de confusion seule

# --- 5. Importance des caractéristiques ---
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B09", "B10", "B11", "B12", "B8A"]

# TODO 7 : construire la liste des 26 noms de caractéristiques, dans le même ordre que X :
#          d'abord "B01_mean", ..., "B8A_mean", puis "B01_std", ..., "B8A_std"
feature_names = [f"{f}_mean" for f in BAND_NAMES]+[f"{f}_std" for f in BAND_NAMES]
# print(len(feature_names), feature_names[0], feature_names[12], feature_names[13], feature_names[25]) # print de vérification

# TODO 8 : récupérer l'importance de chaque caractéristique (26 valeurs)
importances = rf.feature_importances_

# Tri croissant : barh dessine la première barre en bas,
# donc la caractéristique la plus importante apparaîtra en haut du graphique
order = np.argsort(importances)

fig2, ax2 = plt.subplots(figsize=(8, 9))
ax2.barh(np.array(feature_names)[order], importances[order])
ax2.set_xlabel("Mean decrease in impurity")
ax2.set_title("Random Forest – feature importance")
fig2.tight_layout()

out_path2 = Path("figures/rf_feature_importance.png")
fig2.savefig(out_path2, dpi=150)
print("Figure enregistrée dans", out_path2)

plt.show()  # affiche les deux figures