"""Baseline Random Forest sur des caractéristiques spectrales simples (EuroSAT)."""

import time
from pathlib import Path

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
plt.show()