import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from loss_function import AsymmetricLoss
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
TRAIN_PATH = os.path.join(DATA_DIR, "train_complete.parquet")
LABEL_PATH = os.path.join(DATA_DIR, "label.parquet")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "align_model.pth")

BATCH_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProteinGODataset(Dataset):
    def __init__(self, train_df, label_df):
        self.train_df = train_df
        self.go_to_idx = {go_id: idx for idx, go_id in enumerate(label_df["name"])}
        self.num_classes = len(label_df)

        print("Pre-loading training embeddings to RAM...")
        self.embeddings = torch.tensor(
            np.stack(train_df["embedding"].values), dtype=torch.float32
        )
        self.go_terms_list = train_df["go_terms"].values

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        prot_emb = self.embeddings[idx]
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

        terms = self.go_terms_list[idx]
        if terms is not None and len(terms) > 0:
            indices = [self.go_to_idx[t] for t in terms if t in self.go_to_idx]
            if indices:
                label_vec[indices] = 1.0

        return prot_emb, label_vec


def train():
    print(f"Using device: {DEVICE}")

    # Load data
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)

    print("Sorting Labels...")
    label_df = pd.read_parquet(LABEL_PATH).sort_values("name").reset_index(drop=True)

    print(f"Total samples: {len(train_df)}")
    print(f"Total GO terms: {len(label_df)}")

    # Dataset
    full_dataset = ProteinGODataset(train_df, label_df)

    # Split 99/1
    train_size = int(0.99 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

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
    go_embeddings_tensor = torch.tensor(
        np.stack(label_df["embedding"].values), dtype=torch.float32
    ).to(DEVICE)

    # Model (Sử dụng version mới nhất bạn đã sửa: Asymmetric, no go_projector)
    model = ProteinGOAligner(esm_dim=2560, go_emb_dim=768).to(DEVICE)

    # Loss & Optimizer
    # pos_weight = torch.tensor([20.0]).to(DEVICE)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # --- [NEW SCHEDULER: Cosine with Warmup] ---
    warmup_epochs = 2

    # 1. Warmup: Tăng tuyến tính từ LR rất nhỏ lên LR gốc trong 2 epoch
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )

    # 2. Cosine Decay: Giảm theo hình sin từ epoch 3 đến 100
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(EPOCHS - warmup_epochs), eta_min=1e-6
    )

    main_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.0001,
        total_iters=(EPOCHS - warmup_epochs),
    )

    # 3. Kết hợp tuần tự
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    # -------------------------------------------

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0

    print("Starting training (Cosine Schedule with 2 Epochs Warmup)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for prot_emb, labels in progress_bar:
            prot_emb = prot_emb.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            logits = model(prot_emb, go_embeddings_tensor)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for prot_emb, labels in val_loader:
                prot_emb = prot_emb.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(prot_emb, go_embeddings_tensor)
                val_loss += criterion(logits, labels).item()

        avg_val_loss = val_loss / len(val_loader)

        # [MODIFIED] Scheduler step gọi ở cuối epoch, không cần val_loss
        scheduler.step()
        updated_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {updated_lr:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Saved Best Model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    train()
