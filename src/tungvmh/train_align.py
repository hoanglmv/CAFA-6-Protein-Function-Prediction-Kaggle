import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# Add src to path to import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from align_embed.protein_go_aligner import ProteinGOAligner

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed2")
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
LABEL_PATH = os.path.join(DATA_DIR, "label.parquet")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "align_model.pth")
BATCH_SIZE = 512
EPOCHS = 80
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProteinGODataset(Dataset):
    def __init__(self, train_df, label_df):
        self.train_df = train_df
        self.label_df = label_df

        # Create mapping from GO ID to index
        self.go_to_idx = {go_id: idx for idx, go_id in enumerate(label_df["name"])}
        self.num_classes = len(label_df)

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        row = self.train_df.iloc[idx]

        # Protein Embedding
        prot_emb = torch.tensor(row["embedding"], dtype=torch.float32)

        # Multi-hot label vector
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

        go_terms = row["go_terms"]
        if go_terms is not None and isinstance(go_terms, (list, np.ndarray)):
            for go_id in go_terms:
                if go_id in self.go_to_idx:
                    label_vec[self.go_to_idx[go_id]] = 1.0

        return prot_emb, label_vec


def train():
    print(f"Using device: {DEVICE}")

    # Load data
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    label_df = pd.read_parquet(LABEL_PATH)

    print(f"Total samples: {len(train_df)}")
    print(f"Total GO terms: {len(label_df)}")

    # Dataset
    full_dataset = ProteinGODataset(train_df, label_df)

    # Split 90/10
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Prepare GO embeddings
    print("Preparing GO embeddings...")
    go_embeddings_list = label_df["embedding"].tolist()
    go_embeddings_tensor = torch.tensor(
        np.array(go_embeddings_list), dtype=torch.float32
    ).to(DEVICE)

    model = ProteinGOAligner(esm_dim=2560, go_emb_dim=768, joint_dim=512).to(DEVICE)

    # Loss & Optimizer
    # ========================================
    pos_weight = torch.full((full_dataset.num_classes,), 20.0).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # ========================================

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training Loop
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for prot_emb, labels in progress_bar:
            prot_emb = prot_emb.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(prot_emb, go_embeddings_tensor)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for prot_emb, labels in val_loader:
                prot_emb = prot_emb.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(prot_emb, go_embeddings_tensor)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Update Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}"
        )

        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Validation loss improved. Model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
            )

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Run a short training loop for testing"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run mode enabled. Reducing epochs.")
        EPOCHS = 1

    train()
