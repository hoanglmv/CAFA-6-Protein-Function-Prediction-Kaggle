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

BATCH_SIZE = 512
EPOCHS = 40
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5  # Cross-Validation Folds

# --- PARAMETERS ---
EMBEDDING_DIM = 2560
TAXONOMY_DIM = 4  # Kích thước vector Superkingdom
INPUT_DIM = EMBEDDING_DIM + TAXONOMY_DIM  # 2564

# 
# Data Pipeline: Disk -> CPU RAM (Dataset) -> Pinned RAM (DataLoader) -> GPU VRAM (Training)
# Code này giữ dataset ở RAM (vì 500k sample x 2560 float32 ~ 5GB có thể tràn VRAM nếu load hết lên GPU một lúc)
# nhưng luồng train diễn ra hoàn toàn trên GPU.

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
        tax_list = np.stack(train_df["superkingdom"].values)

        # 3. Concatenate Features: [Embedding, Taxonomy] -> (N, 2564)
        print(f"Concatenating embeddings ({EMBEDDING_DIM}) and taxonomy ({TAXONOMY_DIM})...")
        features = np.hstack([emb_list, tax_list])

        # 4. Convert to Tensor & Share Memory (CPU Side)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.features = self.features.share_memory_()

        # Labels
        self.go_terms_id_list = train_df["go_terms_id"].values

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # Data nằm trên CPU, sẽ được DataLoader đẩy lên GPU theo batch
        input_vec = self.features[idx]

        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        indices = self.go_terms_id_list[idx]

        if indices is not None and len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.copy()
            label_vec[indices] = 1.0

        return input_vec, label_vec


def optimize_threshold_gpu(logits, targets, steps=50):
    """
    TÌM NGƯỠNG TỐI ƯU 100% TRÊN GPU (PyTorch Native)
    Thay thế sklearn.f1_score để không phải chuyển dữ liệu về CPU.
    """
    # Sigmoid trên GPU
    probs = torch.sigmoid(logits)
    
    # Tạo danh sách ngưỡng trên GPU
    thresholds = torch.linspace(0.1, 0.6, steps, device=logits.device)
    
    best_t = 0.3
    best_f1 = 0.0
    
    # Kích thước tensor
    # N = targets.shape[0] (samples), C = targets.shape[1] (classes)
    
    for t in thresholds:
        # Dự đoán nhị phân (Binary Mask) ngay trên GPU
        preds = (probs > t).float()
        
        # Tính Micro F1 Score thủ công bằng phép toán ma trận
        # Micro F1 = 2 * TP / (2 * TP + FP + FN)
        
        # TP: Pred=1 & Target=1
        tp = (preds * targets).sum()
        
        # FP: Pred=1 & Target=0
        fp = (preds * (1 - targets)).sum()
        
        # FN: Pred=0 & Target=1
        fn = ((1 - preds) * targets).sum()
        
        # Tính F1 (thêm epsilon để tránh chia cho 0)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    # .item() để lấy giá trị float python ra khỏi tensor 0-dim
    return best_t.item(), best_f1.item()


def train():
    print(f"🚀 Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 1. Load Vocab
    print(f"Loading vocab from {VOCAB_PATH}...")
    with open(VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)
    num_classes = len(vocab_data["top_terms"])
    print(f"Vocab size: {num_classes}")

    # =========================================================================
    # 2. LOAD & CONCAT LABEL EMBEDDINGS (TEXT + GRAPH) -> GPU
    # =========================================================================
    print(f"Loading label embeddings from {LABEL_PATH}...")
    if not os.path.exists(LABEL_PATH):
        print("Label file missing.")
        return

    label_df = pd.read_parquet(LABEL_PATH).sort_values("id")
    text_embeddings = np.stack(label_df["embedding"].values)

    if "node_embedding" in label_df.columns:
        node_embeddings = np.stack(label_df["node_embedding"].values)
        full_go_embeddings = np.concatenate([text_embeddings, node_embeddings], axis=1)
    else:
        full_go_embeddings = text_embeddings

    # --- QUAN TRỌNG: Đưa Label Embeddings lên GPU vĩnh viễn ---
    go_embeddings_tensor = torch.tensor(full_go_embeddings, dtype=torch.float32).to(DEVICE)
    GO_EMB_DIM = go_embeddings_tensor.shape[1]
    print(f"✅ Final GO Embeddings (GPU): {go_embeddings_tensor.shape}")
    # =========================================================================

    # 3. Load Training Data
    print(f"Loading training data from {TRAIN_PATH}...")
    train_df = pd.read_parquet(TRAIN_PATH)
    
    # Dataset
    full_dataset = ProteinGODataset(train_df, num_classes=num_classes)

    # ==================== CROSS VALIDATION ====================
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"\n{'='*20} FOLD {fold+1}/{N_FOLDS} {'='*20}")

        train_subsampler = Subset(full_dataset, train_idx)
        val_subsampler = Subset(full_dataset, val_idx)

        # DataLoader: pin_memory=True giúp copy từ CPU -> GPU nhanh hơn
        train_loader = DataLoader(
            train_subsampler, batch_size=BATCH_SIZE, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_subsampler, batch_size=BATCH_SIZE, shuffle=False, 
            num_workers=4, pin_memory=True
        )

        model = ProteinGOAligner(
            esm_dim=INPUT_DIM, go_emb_dim=GO_EMB_DIM, num_classes=num_classes
        ).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), 
            epochs=EPOCHS, pct_start=0.1
        )

        best_val_f1 = 0.0
        best_threshold_for_fold = 0.3
        patience = 8
        patience_counter = 0

        fold_save_path = os.path.join(MODEL_SAVE_DIR, f"align_model_fold_{fold}.pth")
        threshold_save_path = os.path.join(MODEL_SAVE_DIR, f"threshold_fold_{fold}.json")

        print("Starting training...")
        for epoch in range(EPOCHS):
            # --- TRAINING LOOP (GPU) ---
            model.train()
            train_loss = 0

            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{EPOCHS}")
            for inputs, labels in pbar:
                # non_blocking=True cho phép tính toán song song với việc truyền data
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                logits = model(inputs, go_embeddings_tensor) # go_embeddings_tensor đã ở GPU
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_train_loss = train_loss / len(train_loader)

            # --- VALIDATION LOOP (GPU) ---
            model.eval()
            val_loss = 0
            
            # Giữ tensor trên GPU, KHÔNG chuyển về CPU
            all_logits = []
            all_targets = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)

                    logits = model(inputs, go_embeddings_tensor)
                    val_loss += criterion(logits, labels).item()

                    all_logits.append(logits)
                    all_targets.append(labels)

            avg_val_loss = val_loss / len(val_loader)

            # Nối tensor lại (vẫn trên GPU VRAM)
            # Lưu ý: Nếu VRAM < 8GB và tập Val quá lớn, bước này có thể gây OOM.
            # Nếu bị OOM ở dòng này, buộc phải hy sinh tốc độ để đưa về CPU.
            all_logits_cat = torch.cat(all_logits)
            all_targets_cat = torch.cat(all_targets)

            # --- OPTIMIZE THRESHOLD (GPU) ---
            best_t_epoch, val_f1_epoch = optimize_threshold_gpu(all_logits_cat, all_targets_cat)

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Fold {fold+1} Epoch {epoch+1} | "
                f"Train: {avg_train_loss:.4f} | "
                f"Val: {avg_val_loss:.4f} | "
                f"Best F1: {val_f1_epoch:.4f} (at t={best_t_epoch:.3f}) | LR: {current_lr:.2e}"
            )

            # Checkpointing
            if val_f1_epoch > best_val_f1:
                best_val_f1 = val_f1_epoch
                best_threshold_for_fold = best_t_epoch
                patience_counter = 0
                
                torch.save(model.state_dict(), fold_save_path)
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