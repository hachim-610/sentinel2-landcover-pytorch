"""Entraînement d'un ResNet-18 sur EuroSAT (13 bandes Sentinel-2), avec validation."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT

from s2lc.data import Normalize, load_stats
from s2lc.model import build_model

# --- Réglages ---
N_EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
CLIP = 5
NUM_WORKERS = 4
SEED = 42
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Entraîne le modèle pendant une époque. Renvoie la perte et la précision moyennes."""
    model.train()
    total_loss, n_correct, n_seen = 0.0, 0, 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]
        n_correct += (logits.argmax(dim=1) == labels).sum().item()
        n_seen += images.shape[0]

    return total_loss / n_seen, n_correct / n_seen


def evaluate(model, loader, criterion, device):
    """Évalue le modèle sans l'entraîner. Renvoie la perte et la précision moyennes."""
    # TODO 1 : passer le modèle en mode évaluation
    model.eval()

    total_loss, n_correct, n_seen = 0.0, 0, 0

    # TODO 2 : désactiver le calcul des gradients pour tout le bloc indenté ci-dessous
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.shape[0]
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_seen += images.shape[0]

    return total_loss / n_seen, n_correct / n_seen


def plot_history(history, out_path):
    """Trace les courbes de perte et de précision (entraînement et validation)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], marker="o", label="train")
    ax1.plot(epochs, history["val_loss"], marker="o", label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], marker="o", label="train")
    ax2.plot(epochs, history["val_acc"], marker="o", label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle("ResNet-18 trained from scratch on EuroSAT (13 bands)")
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print("Courbes enregistrées dans", out_path)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    # --- Données ---
    mean, std = load_stats("configs/eurosat_stats.json")
    normalize = Normalize(mean, std, clip=CLIP)
    train_ds = EuroSAT(root="data/eurosat", split="train", transforms=normalize, download=True)
    val_ds = EuroSAT(root="data/eurosat", split="val", transforms=normalize, download=True)

    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, **loader_kwargs)

    # --- Modèle, perte, optimiseur ---
    model = build_model(in_channels=13, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Entraînement ---
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)

    for epoch in range(N_EPOCHS):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # TODO 3 : si la précision de validation bat le record,
        #          mettre à jour le record et sauvegarder les poids
        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            saved = "  <- meilleur modèle sauvegardé"

        print(f"Époque {epoch + 1:2d}/{N_EPOCHS} | {time.time() - start:4.0f} s "
              f"| train : perte {train_loss:.4f}, préc. {train_acc:.3f} "
              f"| val : perte {val_loss:.4f}, préc. {val_acc:.3f}{saved}")

    print(f"\nMeilleure précision de validation : {best_val_acc:.3f}")
    plot_history(history, Path("figures/learning_curves.png"))


if __name__ == "__main__":
    main()