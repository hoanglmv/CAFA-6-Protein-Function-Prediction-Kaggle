import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
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

    print(f"Train samples: {len(train_df)}")
    print(f"Total GO terms: {len(label_df)}")

    # Dataset & DataLoader
    dataset = ProteinGODataset(train_df, label_df)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Prepare GO embeddings
    print("Preparing GO embeddings...")
    go_embeddings_list = label_df["embedding"].tolist()
    go_embeddings_tensor = torch.tensor(
        np.array(go_embeddings_list), dtype=torch.float32
    ).to(DEVICE)

    model = ProteinGOAligner(esm_dim=2560, go_emb_dim=768, joint_dim=512).to(DEVICE)

    # Loss & Optimizer
    # ========================================
    # Added pos_weight as requested
    pos_weight = torch.full((dataset.num_classes,), 20.0).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # ========================================

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for prot_emb, labels in progress_bar:
            prot_emb = prot_emb.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(prot_emb, go_embeddings_tensor)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")


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
