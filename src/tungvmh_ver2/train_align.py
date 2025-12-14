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


# ==================== ASYMMETRIC LOSS ====================
class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


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
        # Pass num_classes for Bias initialization
        print(
            f"Initializing model with Input Dim: {INPUT_DIM} (Protein) and GO Dim: {GO_EMB_DIM} (Label)"
        )
        model = ProteinGOAligner(
            esm_dim=INPUT_DIM, go_emb_dim=GO_EMB_DIM, num_classes=num_classes
        ).to(DEVICE)

        # Loss Function: Asymmetric Loss
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=EPOCHS,
            pct_start=0.1,
        )

        best_val_loss = float(
            "inf"
        )  # Keep track for potential future use, though F1 is primary
        best_val_f1 = 0.0
        patience = 8
        patience_counter = 0

        fold_save_path = os.path.join(MODEL_SAVE_DIR, f"align_model_fold_{fold}.pth")

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
                f"Fold {fold+1} Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | F1: {val_f1:.4f} | LR: {current_lr:.2e}"
            )

            # Checkpointing based on F1 Score (Better for imbalance)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), fold_save_path)
                print(f">>> Saved Best F1 Model for Fold {fold+1}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print(f"Finished Fold {fold+1}. Best F1: {best_val_f1:.4f}")

    print("Cross-Validation Complete!")


if __name__ == "__main__":
    train()
