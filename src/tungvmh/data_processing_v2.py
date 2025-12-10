import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
    DATA_DIR, "processed"
)  # Nơi chứa file metadata (GO terms, Seq) từ data_processing.py
PROCESSED2_DIR = os.path.join(DATA_DIR, "processed2")  # Nơi chứa file kết quả cuối cùng


def get_or_create_embeddings(df_meta, embedding_file_path, desc="Computing Embeddings"):
    """
    Hàm này quản lý file embedding riêng biệt (cache).
    1. Load embedding cache nếu có.
    2. Tìm các ID trong df_meta chưa có trong cache.
    3. Tính embedding cho các ID mới.
    4. Cập nhật cache file.
    5. Trả về DataFrame chứa ID và Embedding (đã merge đủ).
    """

    # 1. Load Cache
    cache_dict = {}
    if os.path.exists(embedding_file_path):
        print(f"Loading embedding cache from {embedding_file_path}...")
        try:
            df_cache = pd.read_parquet(embedding_file_path)
            # Giả sử file cache có cột: id, embedding, superkingdom (optional)
            # Check columns
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
    # Chỉ tính toán cho các ID unique
    unique_ids_in_meta = df_meta["id"].unique()

    missing_ids = [pid for pid in unique_ids_in_meta if pid not in cache_dict]

    if missing_ids:
        print(f"Found {len(missing_ids)} new proteins to embed.")

        # Tạo lookup seq cho nhanh
        id_to_seq = pd.Series(df_meta.seq.values, index=df_meta.id).to_dict()

        for pid in tqdm(missing_ids, desc=desc):
            seq = id_to_seq.get(pid, "")

            # Compute Embedding
            emb_vector = None
            try:
                if not seq:
                    # Nếu không có seq, thử tìm trong cache cũ hoặc skip
                    # Ở đây ta assume seq phải có trong df_meta
                    pass
                else:
                    emb_vector = get_protein_embedding(seq)
            except:
                pass

            if emb_vector is None:
                emb_vector = np.zeros(2560)

            # Superkingdom Placeholder (update logic if needed)
            sk_vector = [0, 0, 0, 0]

            # Update RAM dict
            cache_dict[pid] = {"embedding": emb_vector, "superkingdom": sk_vector}

            # Add to list for saving
            new_entries.append(
                {"id": pid, "embedding": emb_vector, "superkingdom": sk_vector}
            )

        # 4. Update Cache File (Append mode or Rewrite)
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

    # 5. Prepare Output (Merge Metadata with Cache)
    # Chúng ta map embedding từ cache_dict vào df_meta để đảm bảo đúng thứ tự dòng
    print("Merging embeddings into main dataframe...")

    # Tạo list theo thứ tự của df_meta
    emb_list = []
    sk_list = []

    for pid in tqdm(df_meta["id"], desc="Merging"):
        data = cache_dict.get(pid)
        if data:
            emb_list.append(data["embedding"])
            sk_list.append(data["superkingdom"])
        else:
            # Should not happen if logic is correct
            emb_list.append(np.zeros(2560))
            sk_list.append([0, 0, 0, 0])

    # Tạo DataFrame kết quả (Copy để không ảnh hưởng df gốc)
    df_result = df_meta.copy()
    df_result["embedding"] = emb_list
    df_result["superkingdom"] = sk_list

    return df_result


def main():
    os.makedirs(PROCESSED2_DIR, exist_ok=True)

    # --- PROCESS TRAIN ---
    train_meta_path = os.path.join(PROCESSED_DIR, "train.parquet")
    train_emb_path = os.path.join(
        PROCESSED2_DIR, "train_embedding.parquet"
    )  # Cache file
    train_final_path = os.path.join(PROCESSED2_DIR, "train_complete.parquet")

    if os.path.exists(train_meta_path):
        print("\n[TRAIN] Processing...")
        df_train_meta = pd.read_parquet(train_meta_path)

        # Check if cache exists
        if not os.path.exists(train_emb_path):
            print(
                f"Warning: Cache file {train_emb_path} not found. Embeddings will be computed from scratch (SLOW)."
            )
            # Nếu user đã chạy extract_embeddings_temp.py thì file này phải có.

        # Hàm này sẽ tự lo việc load cache, tính cái mới, lưu cache, và merge
        df_train_final = get_or_create_embeddings(
            df_train_meta, train_emb_path, desc="Embedding Train"
        )

        print(f"Saving final train data to {train_final_path}...")
        df_train_final.to_parquet(train_final_path, index=False)
    else:
        print(f"[TRAIN] Metadata file not found: {train_meta_path}")

    # --- PROCESS TEST ---
    test_meta_path = os.path.join(PROCESSED_DIR, "test.parquet")
    test_emb_path = os.path.join(PROCESSED2_DIR, "test_embedding.parquet")  # Cache file
    test_final_path = os.path.join(PROCESSED2_DIR, "test.parquet")

    if os.path.exists(test_meta_path):
        print("\n[TEST] Processing...")
        df_test_meta = pd.read_parquet(test_meta_path)

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
