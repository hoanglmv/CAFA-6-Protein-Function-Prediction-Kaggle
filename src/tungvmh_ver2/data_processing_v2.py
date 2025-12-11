import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.encode.encode_prot import get_protein_embedding
except ImportError as e:
    print(f"Error importing modules: {e}")

    # Fallback function
    def get_protein_embedding(seq):
        return np.zeros(2560)


# Define paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(
    DATA_DIR, "processed_ver2"
)  # Input from data_processing.py
PROCESSED2_DIR = os.path.join(DATA_DIR, "processed2_ver2")  # Output with embeddings


def get_or_create_embeddings(df_meta, embedding_file_path, desc="Computing Embeddings"):
    """
    Manage embedding cache.
    1. Load cache.
    2. Identify missing IDs.
    3. Compute embeddings for missing IDs.
    4. Update cache.
    5. Return merged DataFrame.
    """

    # 1. Load Cache
    cache_dict = {}
    if os.path.exists(embedding_file_path):
        print(f"Loading embedding cache from {embedding_file_path}...")
        try:
            df_cache = pd.read_parquet(embedding_file_path)
            has_sk = "superkingdom" in df_cache.columns

            for _, row in tqdm(
                df_cache.iterrows(), total=len(df_cache), desc="Reading Cache"
            ):
                cache_dict[row["id"]] = {
                    "embedding": row["embedding"],
                    "superkingdom": row["superkingdom"] if has_sk else [0, 0, 0, 0],
                }
        except Exception as e:
            print(f"Error reading cache: {e}. Starting fresh.")

    # 2. Identify Missing
    new_entries = []
    unique_ids_in_meta = df_meta["id"].unique()

    missing_ids = [pid for pid in unique_ids_in_meta if pid not in cache_dict]

    if missing_ids:
        print(f"Found {len(missing_ids)} new proteins to embed.")

        id_to_seq = pd.Series(df_meta.seq.values, index=df_meta.id).to_dict()

        for pid in tqdm(missing_ids, desc=desc):
            seq = id_to_seq.get(pid, "")

            # Compute Embedding
            emb_vector = None
            try:
                if not seq:
                    pass
                else:
                    emb_vector = get_protein_embedding(seq)
            except:
                pass

            if emb_vector is None:
                emb_vector = np.zeros(2560)

            sk_vector = [0, 0, 0, 0]

            # Update RAM dict
            cache_dict[pid] = {"embedding": emb_vector, "superkingdom": sk_vector}

            # Add to list for saving
            new_entries.append(
                {"id": pid, "embedding": emb_vector, "superkingdom": sk_vector}
            )

        # 4. Update Cache File
        if new_entries:
            df_new = pd.DataFrame(new_entries)
            print(f"Updating cache file {embedding_file_path}...")
            if not os.path.exists(embedding_file_path):
                df_new.to_parquet(embedding_file_path, index=False)
            else:
                import pyarrow.parquet as pq
                import pyarrow as pa

                table_new = pa.Table.from_pandas(df_new)
                table_existing = pq.read_table(embedding_file_path)
                table_combined = pa.concat_tables([table_existing, table_new])
                pq.write_table(table_combined, embedding_file_path)
    else:
        print("All embeddings found in cache.")

    # 5. Prepare Output
    print("Merging embeddings into main dataframe...")

    emb_list = []
    sk_list = []

    for pid in tqdm(df_meta["id"], desc="Merging"):
        data = cache_dict.get(pid)
        if data:
            emb_list.append(data["embedding"])
            sk_list.append(data["superkingdom"])
        else:
            emb_list.append(np.zeros(2560))
            sk_list.append([0, 0, 0, 0])

    df_result = df_meta.copy()
    df_result["embedding"] = emb_list
    df_result["superkingdom"] = sk_list

    return df_result


def main():
    os.makedirs(PROCESSED2_DIR, exist_ok=True)

    # --- PROCESS TRAIN ---
    train_meta_path = os.path.join(PROCESSED_DIR, "train.parquet")
    # Reuse existing cache if possible, or create new one in ver2 folder
    # To save time, we can point to the old cache if user wants, but for safety let's create new cache file
    # or we can copy the old cache to the new location.
    # For now, let's use a new cache file path in PROCESSED2_DIR
    train_emb_path = os.path.join(PROCESSED2_DIR, "train_embedding.parquet")
    train_final_path = os.path.join(PROCESSED2_DIR, "train_complete.parquet")

    if os.path.exists(train_meta_path):
        print("\n[TRAIN] Processing...")
        df_train_meta = pd.read_parquet(train_meta_path)

        # Check if we can leverage old cache
        old_cache_path = os.path.join(DATA_DIR, "processed2", "train_embedding.parquet")
        if not os.path.exists(train_emb_path) and os.path.exists(old_cache_path):
             print(f"Copying existing cache from {old_cache_path}...")
             import shutil
             shutil.copy(old_cache_path, train_emb_path)

        df_train_final = get_or_create_embeddings(
            df_train_meta, train_emb_path, desc="Embedding Train"
        )

        print(f"Saving final train data to {train_final_path}...")
        df_train_final.to_parquet(train_final_path, index=False)
    else:
        print(f"[TRAIN] Metadata file not found: {train_meta_path}")

    # --- PROCESS TEST ---
    test_meta_path = os.path.join(PROCESSED_DIR, "test.parquet")
    test_emb_path = os.path.join(PROCESSED2_DIR, "test_embedding.parquet")
    test_final_path = os.path.join(PROCESSED2_DIR, "test.parquet")

    if os.path.exists(test_meta_path):
        print("\n[TEST] Processing...")
        df_test_meta = pd.read_parquet(test_meta_path)

        old_test_cache = os.path.join(DATA_DIR, "processed2", "test_embedding.parquet")
        if not os.path.exists(test_emb_path) and os.path.exists(old_test_cache):
             print(f"Copying existing cache from {old_test_cache}...")
             import shutil
             shutil.copy(old_test_cache, test_emb_path)

        df_test_final = get_or_create_embeddings(
            df_test_meta, test_emb_path, desc="Embedding Test"
        )

        print(f"Saving final test data to {test_final_path}...")
        df_test_final.to_parquet(test_final_path, index=False)
    else:
        print(f"[TEST] Metadata file not found: {test_meta_path}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
