import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import networkx as nx
import obonet
from tqdm import tqdm

# Add src to path to import model
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

# [CẤU HÌNH DIAMOND]
DIAMOND_RAW_PATH = os.path.join(DATA_DIR, "diamond_matches.tsv")
TRAIN_TERMS_PATH = os.path.join(PROJECT_ROOT, "data/Train/train_terms.tsv")

TOP_K = 100 
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.5 

def get_propagation_steps(obo_path, term_to_idx):
    print(f"Loading OBO graph from {obo_path}...")
    graph = obonet.read_obo(obo_path)
    
    valid_terms = set(term_to_idx.keys())
    subgraph = graph.subgraph(valid_terms)
    
    print("Sorting graph topologically...")
    try:
        topo_order = list(nx.topological_sort(subgraph))
    except nx.NetworkXUnfeasible:
        print("Cycle detected, fallback to arbitrary order.")
        topo_order = list(subgraph.nodes())
        
    propagation_steps = []
    for child in tqdm(topo_order, desc="Building Propagation Map"):
        if child not in term_to_idx: continue
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
        print(f"Warning: Diamond file not found at {diamond_file}. Using DL model only.")
        return None
    
    print("Processing Diamond results for Ensemble...")
    
    # 1. Load Diamond Matches
    print("  Loading matches...")
    # Cột 0: TestID, Cột 1: TrainID, Cột 2: Pident
    matches = pd.read_csv(diamond_file, sep='\t', usecols=[0, 1, 2], names=['TestID', 'TrainID', 'Pident'])
    
    # --- [CRITICAL FIX] XỬ LÝ ID TRAIN (sp|ID|NAME -> ID) ---
    print("  Cleaning Train IDs (Removing 'sp|' prefix)...")
    # Tách chuỗi bằng '|' và lấy phần tử thứ 2 (index 1)
    # Ví dụ: "sp|A0A0C5B5G6|MOTSC_HUMAN" -> "A0A0C5B5G6"
    matches['TrainID'] = matches['TrainID'].apply(lambda x: x.split('|')[1] if isinstance(x, str) and '|' in x else str(x))
    # -------------------------------------------------------

    # Filter: Chỉ giữ lại các protein Test có trong tập test hiện tại
    valid_test_ids = set(test_ids)
    matches = matches[matches['TestID'].isin(valid_test_ids)]
    
    # Normalize Score: 100 -> 1.0
    matches['Score'] = matches['Pident'] / 100.0
    
    # 2. Load Ground Truth Labels
    print("  Loading train labels...")
    train_terms = pd.read_csv(train_terms_file, sep='\t', usecols=['EntryID', 'term'])
    train_terms.columns = ['TrainID', 'GoID']
    
    # 3. Merge & Map
    print("  Merging Homology data...")
    # Giờ đây 2 cột TrainID đã khớp format, merge sẽ chạy đúng
    merged = matches.merge(train_terms, on='TrainID', how='inner')
    
    if len(merged) == 0:
        print("WARNING: Merge resulted in 0 records! Check IDs again.")
        return None

    # Group by TestID + GO -> Max Score
    print(f"  Grouping scores from {len(merged)} matches...")
    diamond_preds = merged.groupby(['TestID', 'GoID'])['Score'].max().reset_index()
    
    # 4. Convert to Dictionary
    print("  Building lookup dictionary...")
    diamond_scores_map = {}
    
    for row in tqdm(diamond_preds.itertuples(index=False), total=len(diamond_preds), desc="Mapping Diamond"):
        tid = row.TestID
        goid = row.GoID
        score = row.Score
        
        if goid in term_to_idx:
            go_idx = term_to_idx[goid]
            if tid not in diamond_scores_map:
                diamond_scores_map[tid] = {}
            diamond_scores_map[tid][go_idx] = score
            
    return diamond_scores_map

def predict():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    print("Loading data...")
    test_df = pd.read_parquet(TEST_PATH)
    label_df = pd.read_parquet(LABEL_PATH)
    
    valid_terms_list = label_df["name"].tolist()
    term_to_idx = {term: idx for idx, term in enumerate(valid_terms_list)}
    
    propagation_steps = get_propagation_steps(OBO_PATH, term_to_idx)

    # 2. Load & Process Diamond Scores (ON THE FLY)
    diamond_data = load_and_process_diamond(
        DIAMOND_RAW_PATH, 
        TRAIN_TERMS_PATH, 
        test_df['id'].values, 
        term_to_idx
    )

    # 3. Load Model
    print("Loading model...")
    go_embeddings = torch.tensor(np.stack(label_df["embedding"].tolist())).float().to(DEVICE)
    
    model = ProteinGOAligner(esm_dim=2560, go_emb_dim=768, joint_dim=512).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model.eval()

    # 4. Prediction Loop
    results = []
    num_batches = (len(test_df) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Starting prediction with Ensemble (Alpha={ALPHA})...")
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Predicting"):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(test_df))
            batch_df = test_df.iloc[start_idx:end_idx]
            batch_ids = batch_df["id"].values
            
            prot_emb = torch.tensor(np.stack(batch_df["embedding"].tolist())).float().to(DEVICE)

            # DL Inference
            logits = model(prot_emb, go_embeddings)
            dl_probs = torch.sigmoid(logits).cpu().numpy()

            # Ensemble Logic
            final_probs = dl_probs
            
            if diamond_data:
                diamond_batch = np.zeros_like(dl_probs)
                has_diamond_info = False
                
                for b_idx, pid in enumerate(batch_ids):
                    if pid in diamond_data:
                        has_diamond_info = True
                        for go_idx, score in diamond_data[pid].items():
                            diamond_batch[b_idx, go_idx] = score
                
                if has_diamond_info:
                    final_probs = (ALPHA * dl_probs) + ((1 - ALPHA) * diamond_batch)

            # Propagation
            for child_idx, parent_indices in propagation_steps:
                child_scores = final_probs[:, child_idx:child_idx+1]
                current_parents = final_probs[:, parent_indices]
                new_parent_scores = np.maximum(current_parents, child_scores)
                final_probs[:, parent_indices] = new_parent_scores

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

    print(f"Writing {len(results)} predictions to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        for line in results:
            f.write(line + "\n")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for DL model (0.0 - 1.0)")
    parser.add_argument("--limit", type=int, default=TOP_K, help="Top K predictions")
    args = parser.parse_args()
    
    TOP_K = args.limit
    ALPHA = args.alpha

    predict()