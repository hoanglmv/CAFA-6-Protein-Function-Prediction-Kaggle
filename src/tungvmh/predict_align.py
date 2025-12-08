import os
import torch
import torch.nn as nn
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
TEST_PATH = os.path.join(DATA_DIR, "test.parquet")
LABEL_PATH = os.path.join(DATA_DIR, "label.parquet")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/align_model.pth")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/submission.tsv")
BATCH_SIZE = 8
TOP_K = 15  # Number of predictions to keep per protein
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    print("Loading data...")
    test_df = pd.read_parquet(TEST_PATH)
    label_df = pd.read_parquet(LABEL_PATH)

    print(f"Test samples: {len(test_df)}")
    print(f"Total GO terms: {len(label_df)}")

    # 2. Prepare Model
    print("Loading model...")
    # Load GO embeddings tensor
    go_embeddings_list = label_df["embedding"].tolist()
    go_embeddings_tensor = torch.tensor(
        np.array(go_embeddings_list), dtype=torch.float32
    ).to(DEVICE)

    model = ProteinGOAligner(esm_dim=2560, go_emb_dim=768, joint_dim=512).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    model.eval()

    # 3. Prediction Loop
    results = []

    # Process in batches
    num_batches = (len(test_df) + BATCH_SIZE - 1) // BATCH_SIZE

    print("Starting prediction...")
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Predicting"):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(test_df))
            batch_df = test_df.iloc[start_idx:end_idx]

            # Prepare input
            prot_emb_list = batch_df["embedding"].tolist()
            prot_emb = torch.tensor(np.array(prot_emb_list), dtype=torch.float32).to(
                DEVICE
            )

            # Forward pass
            logits = model(prot_emb, go_embeddings_tensor)
            probs = torch.sigmoid(logits)

            # Get top-k predictions
            # values: [Batch, K], indices: [Batch, K]
            topk_probs, topk_indices = torch.topk(probs, k=TOP_K, dim=1)

            # Convert to CPU for processing
            topk_probs = topk_probs.cpu().numpy()
            topk_indices = topk_indices.cpu().numpy()

            batch_ids = batch_df["id"].values

            for j in range(len(batch_df)):
                prot_id = batch_ids[j]

                for k in range(TOP_K):
                    go_idx = topk_indices[j, k]
                    prob = topk_probs[j, k]

                    go_term_id = label_df.iloc[go_idx]["name"]

                    results.append(f"{prot_id}\t{go_term_id}\t{prob:.3f}")

    # 4. Write Submission File
    print(f"Writing results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        # No header based on sample_submission.tsv check
        for line in results:
            f.write(line + "\n")

    print("Done!")


if __name__ == "__main__":
    import argparse

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
