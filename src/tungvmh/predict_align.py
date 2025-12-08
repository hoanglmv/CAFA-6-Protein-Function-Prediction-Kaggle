import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
OBO_PATH = os.path.join(PROJECT_ROOT, "data/Train/go-basic.obo")  # File OBO gốc
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/align_model.pth")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/submission.tsv")

BATCH_SIZE = 32  # Tăng batch size lên chút vì inference tốn ít VRAM
TOP_K = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_propagation_steps(obo_path, term_to_idx):
    """
    Sử dụng obonet và networkx để tạo danh sách lan truyền điểm.
    Trả về danh sách: [(child_idx, [parent_indices]), ...] được sắp xếp theo Topological Order.
    """
    print(f"Loading OBO graph from {obo_path} using obonet...")
    # obonet đọc file và trả về 1 networkx DiGraph
    # Mặc định: Edge đi từ Child -> Parent (is_a relationship)
    graph = obonet.read_obo(obo_path)

    # Chỉ giữ lại các nodes có trong tập label của chúng ta
    valid_terms = set(term_to_idx.keys())

    # Tạo subgraph chỉ chứa các terms hợp lệ
    # Lưu ý: Nếu A -> B -> C mà B không có trong valid_terms thì A sẽ mất liên kết với C.
    # Tuy nhiên, với tập dataset lớn, hầu hết các term quan trọng đều được giữ.
    subgraph = graph.subgraph(valid_terms)

    print("Sorting graph topologically...")
    # Topological sort đảm bảo chúng ta xử lý node Con trước node Cha
    # Nếu A -> B (A là con B), sort sẽ trả về [A, B, ...]
    try:
        topo_order = list(nx.topological_sort(subgraph))
    except nx.NetworkXUnfeasible:
        print("Warning: Graph contains cycles! Fallback to arbitrary order.")
        topo_order = list(subgraph.nodes())

    propagation_steps = []

    # Tạo các bước propagate
    for child in tqdm(topo_order, desc="Building Propagation Steps"):
        if child not in term_to_idx:
            continue

        child_idx = term_to_idx[child]

        # Tìm cha trực tiếp trong subgraph
        # successors trong obonet graph (Child -> Parent) chính là Parents
        parents = list(subgraph.successors(child))

        if parents:
            parent_indices = [term_to_idx[p] for p in parents if p in term_to_idx]
            if parent_indices:
                propagation_steps.append((child_idx, parent_indices))

    return propagation_steps


def predict():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    print("Loading data...")
    test_df = pd.read_parquet(TEST_PATH)
    label_df = pd.read_parquet(LABEL_PATH)

    print(f"Test samples: {len(test_df)}")
    print(f"Total GO terms: {len(label_df)}")

    # 2. Prepare Consistency Enforcement (Optimized)
    valid_terms_list = label_df["name"].tolist()
    term_to_idx = {term: idx for idx, term in enumerate(valid_terms_list)}

    # Dùng hàm mới với obonet
    propagation_steps = get_propagation_steps(OBO_PATH, term_to_idx)
    print(f"Propagation rules created: {len(propagation_steps)} steps")

    # 3. Prepare Model
    print("Loading model...")
    go_embeddings_list = label_df["embedding"].tolist()

    # Sử dụng np.stack để tạo tensor nhanh hơn
    go_embeddings_tensor = torch.tensor(
        np.stack(go_embeddings_list), dtype=torch.float32
    ).to(DEVICE)

    model = ProteinGOAligner(esm_dim=2560, go_emb_dim=768, joint_dim=512).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    model.eval()

    # 4. Prediction Loop
    results = []
    num_batches = (len(test_df) + BATCH_SIZE - 1) // BATCH_SIZE

    print("Starting prediction...")
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Predicting"):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(test_df))
            batch_df = test_df.iloc[start_idx:end_idx]

            # Prepare input
            prot_emb_list = batch_df["embedding"].tolist()
            prot_emb = torch.tensor(np.stack(prot_emb_list), dtype=torch.float32).to(
                DEVICE
            )

            # Forward pass
            logits = model(prot_emb, go_embeddings_tensor)
            probs = torch.sigmoid(logits)

            # --- Consistency Enforcement (Post-processing) ---
            # Chuyển sang CPU/Numpy để xử lý graph nhanh hơn
            probs_np = probs.cpu().numpy()

            # Propagate scores: Child -> Parent
            # Logic: Score(Parent) = max(Score(Parent), Score(Child))
            # Do đã sort topological, ta chỉ cần chạy 1 pass là đủ
            for child_idx, parent_indices in propagation_steps:
                # Lấy điểm của con cho cả batch [Batch_Size, 1]
                child_scores = probs_np[:, child_idx : child_idx + 1]

                # Lấy điểm hiện tại của các cha [Batch_Size, Num_Parents]
                current_parent_scores = probs_np[:, parent_indices]

                # Update max
                new_parent_scores = np.maximum(current_parent_scores, child_scores)
                probs_np[:, parent_indices] = new_parent_scores
            # -------------------------------------------------

            # Get Top-K
            # Chuyển lại tensor để dùng topk của torch (thường nhanh hơn sort numpy)
            probs_final = torch.from_numpy(probs_np)
            topk_probs, topk_indices = torch.topk(probs_final, k=TOP_K, dim=1)

            topk_probs = topk_probs.numpy()
            topk_indices = topk_indices.numpy()
            batch_ids = batch_df["id"].values

            # Formatting Results
            for j in range(len(batch_df)):
                prot_id = batch_ids[j]
                for k in range(TOP_K):
                    go_idx = topk_indices[j, k]
                    prob = topk_probs[j, k]

                    # Chỉ lưu nếu xác suất > ngưỡng nhỏ (ví dụ 0.001) để giảm dung lượng file
                    # CAFA thường yêu cầu > 0
                    if prob > 0.0:
                        go_term_id = valid_terms_list[
                            go_idx
                        ]  # Lookup từ list nhanh hơn iloc
                        results.append(f"{prot_id}\t{go_term_id}\t{prob:.3f}")

    # 5. Write Submission File
    print(f"Writing results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        for line in results:
            f.write(line + "\n")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=TEST_PATH, help="Path to test parquet file")
    parser.add_argument(
        "--output", default=OUTPUT_PATH, help="Path to output submission file"
    )
    parser.add_argument(
        "--limit", type=int, default=TOP_K, help="Number of top predictions to keep"
    )
    args = parser.parse_args()

    TEST_PATH = args.input
    OUTPUT_PATH = args.output
    TOP_K = args.limit

    predict()
