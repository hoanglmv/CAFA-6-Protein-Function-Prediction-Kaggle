import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
import networkx as nx
import obonet

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

TEST_PATH = os.path.join(PROCESSED2_DIR, "test.parquet")
VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab.pkl")

MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models_ver2")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "submission_ver2.tsv")

# Diamond Paths
DIAMOND_RAW_PATH = os.path.join(DATA_DIR, "processed2", "diamond_matches.tsv")
TRAIN_TERMS_PATH = os.path.join(DATA_DIR, "Train", "train_terms.tsv")
OBO_PATH = os.path.join(DATA_DIR, "Train", "go-basic.obo")

BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 100
N_FOLDS = 5  # Must match train.py

# --- PARAMETERS (Must match train.py) ---
EMBEDDING_DIM = 2560
TAXONOMY_DIM = 4
INPUT_DIM = EMBEDDING_DIM + TAXONOMY_DIM  # 2564


def get_propagation_steps(obo_path, term_to_idx):
    print(f"Loading OBO graph for propagation...")
    graph = obonet.read_obo(obo_path)

    # Only consider terms in our vocab
    valid_terms = set(term_to_idx.keys())
    subgraph = graph.subgraph(valid_terms)

    try:
        topo_order = list(nx.topological_sort(subgraph))
    except:
        # Fallback if cycles exist
        topo_order = list(subgraph.nodes())

    propagation_steps = []
    for child in tqdm(topo_order, desc="Building Propagation"):
        if child not in term_to_idx:
            continue
        child_idx = term_to_idx[child]
        parents = list(subgraph.successors(child))
        parent_indices = [term_to_idx[p] for p in parents if p in term_to_idx]
        if parent_indices:
            propagation_steps.append((child_idx, parent_indices))
    return propagation_steps


def load_and_process_diamond(diamond_file, train_terms_file, test_ids, term_to_idx):
    """
    Load Diamond results and map to local indices.
    """
    if not os.path.exists(diamond_file):
        print(f"Warning: Diamond file not found at {diamond_file}")
        return None

    print("Processing Diamond results for Ensemble...")

    matches = pd.read_csv(
        diamond_file, sep="\t", usecols=[0, 1, 2], names=["TestID", "TrainID", "Pident"]
    )

    def clean_train_id(x):
        x = str(x)
        if "|" in x:
            parts = x.split("|")
            if len(parts) >= 2:
                return parts[1]
        return x

    matches["TrainID"] = matches["TrainID"].apply(clean_train_id)

    valid_test_ids = set(test_ids)
    matches = matches[matches["TestID"].isin(valid_test_ids)]
    matches["Score"] = matches["Pident"] / 100.0

    train_terms = pd.read_csv(train_terms_file, sep="\t", usecols=["EntryID", "term"])
    train_terms.columns = ["TrainID", "GoID"]

    print("Merging Homology data...")
    merged = matches.merge(train_terms, on="TrainID", how="inner")

    if len(merged) == 0:
        print("Warning: Diamond merge resulted in 0 rows.")
        return None

    diamond_preds = merged.groupby(["TestID", "GoID"])["Score"].max().reset_index()

    diamond_scores_map = {}
    for row in tqdm(
        diamond_preds.itertuples(index=False),
        total=len(diamond_preds),
        desc="Mapping Diamond",
    ):
        if row.GoID in term_to_idx:
            go_idx = term_to_idx[row.GoID]
            if row.TestID not in diamond_scores_map:
                diamond_scores_map[row.TestID] = {}
            diamond_scores_map[row.TestID][go_idx] = row.Score

    return diamond_scores_map


def predict(alpha=0.5):
    print(f"Using device: {DEVICE}")

    # 1. Load Vocab
    print(f"Loading vocab from {VOCAB_PATH}...")
    with open(VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)

    top_terms = vocab_data["top_terms"]
    term_to_idx = vocab_data["term_to_idx"]
    num_classes = len(top_terms)
    print(f"Vocab size: {num_classes}")

    # =========================================================================
    # 2. PREPARE GO EMBEDDINGS
    # =========================================================================
    LABEL_PATH = os.path.join(PROCESSED2_DIR, "label.parquet")
    print(f"Loading label embeddings from {LABEL_PATH}...")

    if not os.path.exists(LABEL_PATH):
        print(f"Error: Label embedding file not found at {LABEL_PATH}")
        return

    label_df = pd.read_parquet(LABEL_PATH)
    label_df = label_df.sort_values("id")

    if len(label_df) != num_classes:
        print(f"Warning: Label file has {len(label_df)} terms, expected {num_classes}.")

    print("Stacking Text Embeddings...")
    text_embeddings = np.stack(label_df["embedding"].values)

    if "node_embedding" in label_df.columns:
        print("Stacking Graph Node Embeddings...")
        node_embeddings = np.stack(label_df["node_embedding"].values)
        print("🔗 Concatenating Text and Node Embeddings...")
        full_go_embeddings = np.concatenate([text_embeddings, node_embeddings], axis=1)
    else:
        print(
            "⚠️ Warning: 'node_embedding' column not found. Using only text embeddings."
        )
        full_go_embeddings = text_embeddings

    go_embeddings_tensor = torch.tensor(full_go_embeddings, dtype=torch.float32).to(
        DEVICE
    )
    GO_EMB_DIM = go_embeddings_tensor.shape[1]
    print(
        f"✅ Final GO Embeddings Shape: {go_embeddings_tensor.shape} (Dim: {GO_EMB_DIM})"
    )
    # =========================================================================

    # 3. Load Test Data
    print(f"Loading test data from {TEST_PATH}...")
    if not os.path.exists(TEST_PATH):
        print("Test data not found.")
        return

    test_df = pd.read_parquet(TEST_PATH)
    print(f"Total test samples: {len(test_df)}")

    # 4. Prepare Diamond & Propagation
    propagation_steps = get_propagation_steps(OBO_PATH, term_to_idx)
    diamond_data = load_and_process_diamond(
        DIAMOND_RAW_PATH, TRAIN_TERMS_PATH, test_df["id"].values, term_to_idx
    )

    # 5. Load Models (Ensemble K-Fold)
    models = []
    print(f"Loading {N_FOLDS} models for ensemble...")

    for fold in range(N_FOLDS):
        model_path = os.path.join(MODEL_SAVE_DIR, f"align_model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"⚠️ Model for Fold {fold} not found at {model_path}. Skipping.")
            continue

        model = ProteinGOAligner(
            esm_dim=INPUT_DIM, go_emb_dim=GO_EMB_DIM, num_classes=num_classes
        ).to(DEVICE)

        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            models.append(model)
            print(f"✅ Loaded Fold {fold}")
        except Exception as e:
            print(f"❌ Error loading Fold {fold}: {e}")

    if not models:
        print("No models loaded! Exiting.")
        return

    # 6. Predict
    results = []
    num_batches = (len(test_df) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Predicting (Alpha={alpha})...")
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(test_df))
            batch_df = test_df.iloc[start_idx:end_idx]
            batch_ids = batch_df["id"].values

            # --- DATA PREPARATION ---
            emb_batch = np.stack(batch_df["embedding"].values)
            tax_batch = np.stack(batch_df["superkingdom"].values)
            features = np.hstack([emb_batch, tax_batch])
            prot_input = torch.tensor(features).float().to(DEVICE)

            # --- ENSEMBLE PREDICTION ---
            avg_probs = None
            for model in models:
                logits = model(prot_input, go_embeddings_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            dl_probs = avg_probs / len(models)
            final_probs = dl_probs.copy()

            # Ensemble Diamond
            if diamond_data:
                for b_idx, pid in enumerate(batch_ids):
                    if pid in diamond_data:
                        d_scores = np.zeros(dl_probs.shape[1], dtype=np.float32)
                        for go_idx, score in diamond_data[pid].items():
                            if go_idx < dl_probs.shape[1]:
                                d_scores[go_idx] = score

                        final_probs[b_idx] = (alpha * dl_probs[b_idx]) + (
                            (1 - alpha) * d_scores
                        )

            # Propagation
            for child_idx, parent_indices in propagation_steps:
                child_scores = final_probs[:, child_idx : child_idx + 1]
                current_parents = final_probs[:, parent_indices]
                final_probs[:, parent_indices] = np.maximum(
                    current_parents, child_scores
                )

            # Top K
            final_probs_t = torch.from_numpy(final_probs)
            k_val = min(TOP_K, final_probs_t.shape[1])
            topk_vals, topk_inds = torch.topk(final_probs_t, k=k_val, dim=1)

            topk_vals = topk_vals.numpy()
            topk_inds = topk_inds.numpy()

            for j, prot_id in enumerate(batch_ids):
                for k in range(k_val):
                    prob = topk_vals[j, k]
                    if prob > 0.001:
                        idx = topk_inds[j, k]
                        term = top_terms[idx]
                        results.append(f"{prot_id}\t{term}\t{prob:.3f}")

    print(f"Writing results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        for line in results:
            f.write(line + "\n")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for DL model (0.0-1.0)"
    )
    args = parser.parse_args()
    predict(alpha=args.alpha)
