import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Define paths
DATA_DIR = "data/processed2"
TRAIN_COMPLETE_PATH = os.path.join(DATA_DIR, "train_complete.parquet")
TEST_COMPLETE_PATH = os.path.join(DATA_DIR, "test.parquet")

TRAIN_EMB_OUT = os.path.join(DATA_DIR, "train_embedding.parquet")
TEST_EMB_OUT = os.path.join(DATA_DIR, "test_embedding.parquet")


def extract_and_save(input_path, output_path, desc="Extracting"):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)

    # Check required columns
    required_cols = ["id", "embedding"]
    if not all(col in df.columns for col in required_cols):
        print(f"Missing columns in {input_path}. Available: {df.columns}")
        return

    # Extract only necessary columns
    # We keep 'seq' and 'superkingdom' if they exist, as they are part of the "embedding/feature" cache
    cols_to_keep = ["id", "embedding"]
    if "seq" in df.columns:
        cols_to_keep.append("seq")
    if "superkingdom" in df.columns:
        cols_to_keep.append("superkingdom")

    print(f"Extracting {cols_to_keep}...")
    df_extracted = df[cols_to_keep].copy()

    print(f"Saving to {output_path}...")
    df_extracted.to_parquet(output_path, index=False)
    print("Done.")


def main():
    print("--- EXTRACTING EMBEDDINGS ---")
    # Ensure output directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    extract_and_save(TRAIN_COMPLETE_PATH, TRAIN_EMB_OUT, desc="Train Embeddings")
    extract_and_save(TEST_COMPLETE_PATH, TEST_EMB_OUT, desc="Test Embeddings")
    print("\nExtraction complete.")


if __name__ == "__main__":
    main()
