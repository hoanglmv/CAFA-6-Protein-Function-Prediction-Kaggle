import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.metrics import f1_score

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.align_embed.protein_go_aligner import ProteinGOAligner
except ImportError:
    pass

# Configuration
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_ver2")
PROCESSED2_DIR = os.path.join(DATA_DIR, "processed2_ver2")

TRAIN_PATH = os.path.join(PROCESSED2_DIR, "train_complete.parquet")
VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab.pkl")
LABEL_PATH = os.path.join(PROCESSED2_DIR, "label.parquet")

MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models_ver2")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "align_model.pth")

BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PARAMETERS ---
EMBEDDING_DIM = 2560
TAXONOMY_DIM = 4  # Kích thước vector Superkingdom
INPUT_DIM = EMBEDDING_DIM + TAXONOMY_DIM  # 2564


class ProteinGODataset(Dataset):
    def __init__(self, train_df, num_classes):
        self.train_df = train_df
        self.num_classes = num_classes

        print("Pre-loading data to RAM...")

        # 1. Protein Embeddings (N, 2560)
        print("Stacking Embeddings...")
        emb_list = np.stack(train_df["embedding"].values)

        # 2. Superkingdom Vectors (N, 4)
        print("Stacking Taxonomy Vectors...")
        # Đảm bảo cột superkingdom là dạng list/array
        tax_list = np.stack(train_df["superkingdom"].values)

        # 3. Concatenate Features: [Embedding, Taxonomy] -> (N, 2564)
        print(
            f"Concatenating embeddings ({EMBEDDING_DIM}) and taxonomy ({TAXONOMY_DIM})..."
        )
        features = np.hstack([emb_list, tax_list])

        # 4. Convert to Tensor & Share Memory
        self.features = torch.tensor(features, dtype=torch.float32)
        self.features = self.features.share_memory_()  # Optimize for DataLoader workers

        # Labels
        # Lưu ý: Các phần tử trong mảng này có thể là numpy array read-only
        self.go_terms_id_list = train_df["go_terms_id"].values

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # Lấy feature vector đã nối (2564 dim)
        input_vec = self.features[idx]

        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        indices = self.go_terms_id_list[idx]

        if indices is not None and len(indices) > 0:
            # --- FIX WARNING: The given NumPy array is not writable ---
            # PyTorch yêu cầu mảng numpy phải writable khi dùng làm index.
            # Dữ liệu đọc từ Parquet/Pandas đôi khi là read-only view.
            # Ta copy ra bản mới để đảm bảo writable.
            if isinstance(indices, np.ndarray):
                indices = indices.copy()
            # ----------------------------------------------------------

            label_vec[indices] = 1.0

        return input_vec, label_vec


def train():
    print(f"Using device: {DEVICE}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 1. Load Vocab
    print(f"Loading vocab from {VOCAB_PATH}...")
    with open(VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)
    num_classes = len(vocab_data["top_terms"])
    print(f"Vocab size: {num_classes}")

    # 2. Load Label Embeddings
    print(f"Loading label embeddings from {LABEL_PATH}...")
    if not os.path.exists(LABEL_PATH):
        print("Label file missing.")
        return
    label_df = pd.read_parquet(LABEL_PATH).sort_values("id")
    train_go_embeddings = np.stack(label_df["embedding"].values)
    go_embeddings_tensor = torch.tensor(train_go_embeddings, dtype=torch.float32).to(
        DEVICE
    )
    print(f"GO Embeddings Shape: {go_embeddings_tensor.shape}")

    # 3. Load Training Data
    print(f"Loading training data from {TRAIN_PATH}...")
    if not os.path.exists(TRAIN_PATH):
        print("Train data missing.")
        return
    train_df = pd.read_parquet(TRAIN_PATH)
    print(f"Total samples: {len(train_df)}")

    # Dataset & Loader
    full_dataset = ProteinGODataset(train_df, num_classes=num_classes)

    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

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

    # 4. Model Initialization
    # Quan trọng: Cập nhật esm_dim = INPUT_DIM (2564)
    print(
        f"Initializing model with Input Dim: {INPUT_DIM} (ESM: {EMBEDDING_DIM} + Tax: {TAXONOMY_DIM})"
    )
    model = ProteinGOAligner(esm_dim=INPUT_DIM, go_emb_dim=768).to(DEVICE)

    # Loss Function
    # Dùng BCEWithLogitsLoss với pos_weight để xử lý mất cân bằng
    pos_weight = torch.tensor([15.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
    )

    # Training Loop
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    patience = 8
    patience_counter = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs, go_embeddings_tensor)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(inputs, go_embeddings_tensor)
                val_loss += criterion(logits, labels).item()

                # Simple F1 Check (Threshold 0.3)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.3).cpu().int()
                all_preds.append(preds)
                all_targets.append(labels.cpu().int())

        avg_val_loss = val_loss / len(val_loader)

        # Calc Metrics
        all_preds_cat = torch.cat(all_preds).numpy()
        all_targets_cat = torch.cat(all_targets).numpy()
        val_f1 = f1_score(all_targets_cat, all_preds_cat, average="micro")

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | F1: {val_f1:.4f} | LR: {current_lr:.2e}"
        )

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(">>> Saved Best Model (Lowest Loss)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Optional: Save best F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_SAVE_DIR, "align_model_best_f1.pth"),
            )
            print(">>> Saved Best F1 Model")


if __name__ == "__main__":
    train()
