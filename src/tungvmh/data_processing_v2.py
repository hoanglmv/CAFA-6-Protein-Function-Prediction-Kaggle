import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.encode.encode_prot import get_protein_embedding
    from src.map_taxonomy.taxonomy_group import TaxonGrouper
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(
        "Please make sure you are running this script with the project root in PYTHONPATH."
    )
    sys.exit(1)

# Define paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PROCESSED2_DIR = os.path.join(DATA_DIR, "processed2")


def process_dataframe(df, grouper, desc="Processing"):
    """
    Process a dataframe: add superkingdom and embedding columns.
    """
    # Lists to store new columns
    superkingdoms = []
    embeddings = []

    # Iterate with tqdm
    for index, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        # 1. Get Superkingdom One-Hot Vector
        # Ensure taxonomy is a string or int as expected by grouper
        tax_id = row.get("taxonomy")
        if pd.isna(tax_id):
            tax_id = "0"  # Handle missing taxonomy

        # grouper.get_one_hot expects tax_id
        sk_vector = grouper.get_one_hot(tax_id)
        superkingdoms.append(sk_vector)

        # 2. Get Protein Embedding
        seq = row.get("seq", "")
        if not seq:
            # Handle empty sequence if necessary, though unlikely in this dataset
            # Embedding dim depends on model, assume it handles empty or we skip
            # For safety, let's pass it, but usually we expect valid seq
            emb_vector = np.zeros(
                2560
            )  # Placeholder if needed, but better to let model handle or error
            # Actually get_protein_embedding might fail on empty string, let's try
            pass

        try:
            emb_vector = get_protein_embedding(seq)
        except Exception as e:
            print(f"Error encoding sequence for ID {row.get('id')}: {e}")
            emb_vector = None  # Or some default

        embeddings.append(emb_vector)

    # Add columns to dataframe
    # We need to ensure they are numpy arrays
    df["superkingdom"] = list(superkingdoms)
    df["embedding"] = list(embeddings)

    return df


def main():
    # Create output directory
    os.makedirs(PROCESSED2_DIR, exist_ok=True)

    # Initialize TaxonGrouper
    print("Initializing TaxonGrouper...")
    grouper = TaxonGrouper()

    # Process Train Data
    train_input = os.path.join(PROCESSED_DIR, "train.parquet")
    train_output = os.path.join(PROCESSED2_DIR, "train.parquet")

    if os.path.exists(train_input):
        print(f"\nLoading train data from {train_input}...")
        df_train = pd.read_parquet(train_input)

        # Optional: Process a subset for testing if needed
        # df_train = df_train.head(100)

        print("Processing train data...")
        df_train_processed = process_dataframe(df_train, grouper, desc="Train Data")

        print(f"Saving processed train data to {train_output}...")
        df_train_processed.to_parquet(train_output, index=False)
    else:
        print(f"Train input file not found: {train_input}")

    # Process Test Data
    test_input = os.path.join(PROCESSED_DIR, "test.parquet")
    test_output = os.path.join(PROCESSED2_DIR, "test.parquet")

    if os.path.exists(test_input):
        print(f"\nLoading test data from {test_input}...")
        df_test = pd.read_parquet(test_input)

        # Optional: Process a subset for testing
        # df_test = df_test.head(100)

        print("Processing test data...")
        df_test_processed = process_dataframe(df_test, grouper, desc="Test Data")

        print(f"Saving processed test data to {test_output}...")
        df_test_processed.to_parquet(test_output, index=False)
    else:
        print(f"Test input file not found: {test_input}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
