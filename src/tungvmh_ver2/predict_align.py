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
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "align_model_best_f1.pth")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "submission_ver2.tsv")

# Diamond Paths
DIAMOND_RAW_PATH = os.path.join(DATA_DIR, "processed2", "diamond_matches.tsv")
TRAIN_TERMS_PATH = os.path.join(DATA_DIR, "Train", "train_terms.tsv")
OBO_PATH = os.path.join(DATA_DIR, "Train", "go-basic.obo")

BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 100

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
    Load Diamond results and map to local indices (0-4999).
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

    # 2. Prepare GO Embeddings
    LABEL_PATH = os.path.join(PROCESSED2_DIR, "label.parquet")
    print(f"Loading label embeddings from {LABEL_PATH}...")

    if not os.path.exists(LABEL_PATH):
        print(f"Error: Label embedding file not found at {LABEL_PATH}")
        return

    label_df = pd.read_parquet(LABEL_PATH)
    label_df = label_df.sort_values("id")

    if len(label_df) != num_classes:
        print(f"Warning: Label file has {len(label_df)} terms, expected {num_classes}.")

    train_go_embeddings = np.stack(label_df["embedding"].values)

    go_embeddings_tensor = torch.tensor(train_go_embeddings, dtype=torch.float32).to(
        DEVICE
    )

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

    # 5. Load Model
    # UPDATE: Khởi tạo model với INPUT_DIM = 2564
    print(f"Initializing model with esm_dim={INPUT_DIM} (2560 ESM + 4 Tax)...")
    model = ProteinGOAligner(esm_dim=INPUT_DIM, go_emb_dim=768).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    print(f"Loading weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

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

            # --- NEW DATA PREPARATION LOGIC ---
            # 1. Get Embeddings (Batch, 2560)
            emb_batch = np.stack(batch_df["embedding"].values)

            # 2. Get Superkingdom Vectors (Batch, 4)
            # data_processing2.py đã đảm bảo cột này tồn tại trong test.parquet
            tax_batch = np.stack(batch_df["superkingdom"].values)

            # 3. Concatenate -> (Batch, 2564)
            features = np.hstack([emb_batch, tax_batch])

            # 4. Convert to Tensor
            prot_input = torch.tensor(features).float().to(DEVICE)
            # -----------------------------------

            # DL Scores
            logits = model(prot_input, go_embeddings_tensor)
            dl_probs = torch.sigmoid(logits).cpu().numpy()  # (Batch, 5000)

            final_probs = dl_probs

            # Ensemble Diamond
            if diamond_data:
                diamond_batch = np.zeros_like(dl_probs)
                has_info = False
                for b_idx, pid in enumerate(batch_ids):
                    if pid in diamond_data:
                        has_info = True
                        for go_idx, score in diamond_data[pid].items():
                            diamond_batch[b_idx, go_idx] = score

                if has_info:
                    final_probs = (alpha * dl_probs) + ((1 - alpha) * diamond_batch)

            # Propagation
            for child_idx, parent_indices in propagation_steps:
                child_scores = final_probs[:, child_idx : child_idx + 1]
                current_parents = final_probs[:, parent_indices]
                final_probs[:, parent_indices] = np.maximum(
                    current_parents, child_scores
                )

            # Top K
            final_probs_t = torch.from_numpy(final_probs)
            topk_vals, topk_inds = torch.topk(final_probs_t, k=TOP_K, dim=1)

            topk_vals = topk_vals.numpy()
            topk_inds = topk_inds.numpy()

            for j, prot_id in enumerate(batch_ids):
                for k in range(TOP_K):
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
