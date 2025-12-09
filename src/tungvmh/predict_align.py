import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import networkx as nx
import obonet
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from align_embed.protein_go_aligner import ProteinGOAligner

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed2")

TEST_PATH = os.path.join(DATA_DIR, "test.parquet")
LABEL_PATH = os.path.join(DATA_DIR, "label.parquet")
OBO_PATH = os.path.join(PROJECT_ROOT, "data/Train/go-basic.obo")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/align_model.pth")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/submission.tsv")

DIAMOND_RAW_PATH = os.path.join(DATA_DIR, "diamond_matches.tsv")
TRAIN_TERMS_PATH = os.path.join(PROJECT_ROOT, "data/Train/train_terms.tsv")

# Tăng TOP_K lên 100 để tối ưu F-max
TOP_K = 100
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_propagation_steps(obo_path, term_to_idx):
    print(f"Loading OBO graph...")
    graph = obonet.read_obo(obo_path)
    valid_terms = set(term_to_idx.keys())
    subgraph = graph.subgraph(valid_terms)

    try:
        topo_order = list(nx.topological_sort(subgraph))
    except:
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
    Xử lý Diamond Blastp và FIX LỖI ID MISMATCH.
    """
    if not os.path.exists(diamond_file):
        print(f"Warning: Diamond file not found.")
        return None

    print("Processing Diamond results for Ensemble...")

    # 1. Load Diamond Matches
    matches = pd.read_csv(
        diamond_file, sep="\t", usecols=[0, 1, 2], names=["TestID", "TrainID", "Pident"]
    )

    # --- [FIX QUAN TRỌNG: Cắt chuỗi sp|ID|NAME] ---
    def clean_train_id(x):
        x = str(x)
        if "|" in x:
            parts = x.split("|")
            if len(parts) >= 2:
                return parts[1]  # Lấy ID ở giữa
        return x

    matches["TrainID"] = matches["TrainID"].apply(clean_train_id)
    # ----------------------------------------------

    valid_test_ids = set(test_ids)
    matches = matches[matches["TestID"].isin(valid_test_ids)]
    matches["Score"] = matches["Pident"] / 100.0

    # 2. Load Ground Truth Labels
    train_terms = pd.read_csv(train_terms_file, sep="\t", usecols=["EntryID", "term"])
    train_terms.columns = ["TrainID", "GoID"]

    # 3. Merge & Map
    print("Merging Homology data...")
    merged = matches.merge(train_terms, on="TrainID", how="inner")

    print(f"DEBUG: Diamond Raw Rows: {len(matches)}")
    print(f"DEBUG: Merged Rows (Hits): {len(merged)}")

    if len(merged) == 0:
        print("🚨 LỖI: Merge ra 0 dòng! Kiểm tra ID.")
        return None

    # 4. Create Dict
    diamond_preds = merged.groupby(["TestID", "GoID"])["Score"].max().reset_index()

    diamond_scores_map = {}
    for row in tqdm(
        diamond_preds.itertuples(index=False), total=len(diamond_preds), desc="Mapping"
    ):
        if row.GoID in term_to_idx:
            go_idx = term_to_idx[row.GoID]
            if row.TestID not in diamond_scores_map:
                diamond_scores_map[row.TestID] = {}
            diamond_scores_map[row.TestID][go_idx] = row.Score

    return diamond_scores_map


def predict(alpha=0.5):
    print(f"Using device: {DEVICE}")

    test_df = pd.read_parquet(TEST_PATH)
    # Sort label giống hệt lúc train
    label_df = pd.read_parquet(LABEL_PATH).sort_values("name").reset_index(drop=True)

    valid_terms_list = label_df["name"].tolist()
    term_to_idx = {term: idx for idx, term in enumerate(valid_terms_list)}

    propagation_steps = get_propagation_steps(OBO_PATH, term_to_idx)

    # Load Diamond
    diamond_data = load_and_process_diamond(
        DIAMOND_RAW_PATH, TRAIN_TERMS_PATH, test_df["id"].values, term_to_idx
    )

    # Load Model
    go_embeddings = (
        torch.tensor(np.stack(label_df["embedding"].tolist())).float().to(DEVICE)
    )
    model = ProteinGOAligner(esm_dim=2560, go_emb_dim=768, joint_dim=512).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    results = []
    num_batches = (len(test_df) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Predicting (Alpha={alpha})...")
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(test_df))
            batch_df = test_df.iloc[start_idx:end_idx]
            batch_ids = batch_df["id"].values

            prot_emb = (
                torch.tensor(np.stack(batch_df["embedding"].tolist()))
                .float()
                .to(DEVICE)
            )

            # DL Score
            logits = model(prot_emb, go_embeddings)
            dl_probs = torch.sigmoid(logits).cpu().numpy()

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

            # Top-K
            final_probs_t = torch.from_numpy(final_probs)
            topk_vals, topk_inds = torch.topk(final_probs_t, k=TOP_K, dim=1)

            topk_vals = topk_vals.numpy()
            topk_inds = topk_inds.numpy()

            for j, prot_id in enumerate(batch_ids):
                for k in range(TOP_K):
                    prob = topk_vals[j, k]
                    if prob > 0.001:
                        go_id = valid_terms_list[topk_inds[j, k]]
                        results.append(f"{prot_id}\t{go_id}\t{prob:.3f}")

    print(f"Writing results...")
    with open(OUTPUT_PATH, "w") as f:
        for line in results:
            f.write(line + "\n")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    predict(alpha=args.alpha)
