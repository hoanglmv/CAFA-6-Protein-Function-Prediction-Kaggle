import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.metrics import f1_score
import json

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

BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5  # Cross-Validation Folds

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
            if isinstance(indices, np.ndarray):
                indices = indices.copy()
            # ----------------------------------------------------------

            label_vec[indices] = 1.0

        return input_vec, label_vec

def optimize_threshold(logits, targets, steps=50):
    """
    Tìm ngưỡng t tối ưu hóa F1 Score (Micro) trên tập validation
    """
    # Chuyển sang sigmoid probability
    probs = torch.sigmoid(logits).cpu().numpy()
    targets = targets.cpu().numpy()
    
    best_t = 0.3 # Default
    best_f1 = 0.0
    
    # Search threshold từ 0.1 đến 0.6
    thresholds = np.linspace(0.1, 0.6, steps)
    
    for t in thresholds:
        # Dự đoán
        preds = (probs > t).astype(int)
        
        # Tính F1 Micro (Phù hợp với bài toán multi-label imbalance)
        f1 = f1_score(targets, preds, average="micro")
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    return best_t, best_f1

def train():
    print(f"Using device: {DEVICE}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 1. Load Vocab
    print(f"Loading vocab from {VOCAB_PATH}...")
    with open(VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)
    num_classes = len(vocab_data["top_terms"])
    print(f"Vocab size: {num_classes}")

    # =========================================================================
    # 2. LOAD & CONCAT LABEL EMBEDDINGS (TEXT + GRAPH)
    # =========================================================================
    print(f"Loading label embeddings from {LABEL_PATH}...")
    if not os.path.exists(LABEL_PATH):
        print("Label file missing.")
        return

    label_df = pd.read_parquet(LABEL_PATH).sort_values("id")

    # a. Load Text Embeddings (BioBERT/ESM) - Shape: (Num_Classes, 768)
    print("Stacking Text Embeddings...")
    text_embeddings = np.stack(label_df["embedding"].values)

    # b. Load Node Embeddings (Node2Vec) - Shape: (Num_Classes, 64)
    # Kiểm tra xem cột node_embedding có tồn tại không
    if "node_embedding" in label_df.columns:
        print("Stacking Graph Node Embeddings...")
        node_embeddings = np.stack(label_df["node_embedding"].values)

        # c. Concatenate: [Text, Node] -> (Num_Classes, 768 + 64)
        print("🔗 Concatenating Text and Node Embeddings...")
        full_go_embeddings = np.concatenate([text_embeddings, node_embeddings], axis=1)
    else:
        print(
            "⚠️ Warning: 'node_embedding' column not found. Using only text embeddings."
        )
        full_go_embeddings = text_embeddings

    # Chuyển sang Tensor
    go_embeddings_tensor = torch.tensor(full_go_embeddings, dtype=torch.float32).to(
        DEVICE
    )

    # Tự động lấy kích thước embedding mới
    GO_EMB_DIM = go_embeddings_tensor.shape[1]
    print(
        f"✅ Final GO Embeddings Shape: {go_embeddings_tensor.shape} (Dim: {GO_EMB_DIM})"
    )
    # =========================================================================

    # 3. Load Training Data
    print(f"Loading training data from {TRAIN_PATH}...")
    if not os.path.exists(TRAIN_PATH):
        print("Train data missing.")
        return
    train_df = pd.read_parquet(TRAIN_PATH)
    print(f"Total samples: {len(train_df)}")

    # Dataset
    full_dataset = ProteinGODataset(train_df, num_classes=num_classes)

    # ==================== CROSS VALIDATION ====================
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"\n{'='*20} FOLD {fold+1}/{N_FOLDS} {'='*20}")

        train_subsampler = Subset(full_dataset, train_idx)
        val_subsampler = Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_subsampler,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subsampler,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Model Initialization (Fresh for each fold)
        print(
            f"Initializing model with Input Dim: {INPUT_DIM} (Protein) and GO Dim: {GO_EMB_DIM} (Label)"
        )
        model = ProteinGOAligner(
            esm_dim=INPUT_DIM, go_emb_dim=GO_EMB_DIM, num_classes=num_classes
        ).to(DEVICE)

        # Loss Function: BCE With Logits Loss
        criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=EPOCHS,
            pct_start=0.1,
        )

        best_val_f1 = 0.0
        best_threshold_for_fold = 0.3
        patience = 8
        patience_counter = 0

        fold_save_path = os.path.join(MODEL_SAVE_DIR, f"align_model_fold_{fold}.pth")
        threshold_save_path = os.path.join(MODEL_SAVE_DIR, f"threshold_fold_{fold}.json")

        print("Starting training...")
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0

            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{EPOCHS}")
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

            # ================= VALIDATION & THRESHOLD OPTIMIZATION =================
            model.eval()
            val_loss = 0
            
            # Lưu lại toàn bộ logits và targets để tìm threshold tối ưu
            all_logits = []
            all_targets = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    logits = model(inputs, go_embeddings_tensor)
                    val_loss += criterion(logits, labels).item()

                    all_logits.append(logits)
                    all_targets.append(labels)

            avg_val_loss = val_loss / len(val_loader)

            # Ghép toàn bộ batch lại
            all_logits_cat = torch.cat(all_logits)
            all_targets_cat = torch.cat(all_targets)

            # Tìm ngưỡng tốt nhất cho epoch này
            best_t_epoch, val_f1_epoch = optimize_threshold(all_logits_cat, all_targets_cat)

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Fold {fold+1} Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Best F1: {val_f1_epoch:.4f} (at t={best_t_epoch:.3f}) | LR: {current_lr:.2e}"
            )

            # Checkpointing based on F1 Score
            if val_f1_epoch > best_val_f1:
                best_val_f1 = val_f1_epoch
                best_threshold_for_fold = best_t_epoch # Lưu ngưỡng tốt nhất của fold
                patience_counter = 0
                
                # Save Model
                torch.save(model.state_dict(), fold_save_path)
                
                # Save Best Threshold to JSON
                with open(threshold_save_path, "w") as f:
                    json.dump({"threshold": float(best_threshold_for_fold), "f1": float(best_val_f1)}, f)
                    
                print(f">>> Saved Best Model & Threshold ({best_t_epoch:.3f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print(f"Finished Fold {fold+1}. Best F1: {best_val_f1:.4f} at Threshold: {best_threshold_for_fold:.3f}")

    print("Cross-Validation Complete!")


if __name__ == "__main__":
    train()