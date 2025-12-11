import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from ete3 import NCBITaxa

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
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_ver2")
PROCESSED2_DIR = os.path.join(DATA_DIR, "processed2_ver2")


# ================= SUPERKINGDOM MAPPING LOGIC (ETE3) =================
def build_superkingdom_map(unique_tax_ids):
    """
    Sử dụng thư viện ete3 để tra cứu lineage của từng TaxID
    và map về 4 nhóm chính:
    0: Bacteria
    1: Eukaryota
    2: Archaea
    3: Virus/Other
    """
    print(f"Initializing NCBI Taxonomy database (ete3)...")
    # Lần đầu chạy dòng này sẽ tự download file taxdump.tar.gz (~100MB) từ NCBI
    # và tạo database sqlite cục bộ.
    ncbi = NCBITaxa()

    print(f"Mapping {len(unique_tax_ids)} unique taxonomy IDs to Superkingdoms...")

    # Mapping dict: Name -> Index
    GROUP_INDICES = {"Bacteria": 0, "Eukaryota": 1, "Archaea": 2, "Viruses": 3}

    tax_map = {}

    for tax_id in tqdm(unique_tax_ids, desc="Mapping Taxonomy"):
        # Mặc định là nhóm 3 (Other/Virus)
        vec = [0, 0, 0, 0]
        group_idx = 3

        # Bỏ qua ID 0 hoặc ID âm
        if tax_id <= 0:
            vec[group_idx] = 1
            tax_map[tax_id] = vec
            continue

        try:
            # Lấy toàn bộ cây tổ tiên (lineage) của tax_id
            lineage = ncbi.get_lineage(tax_id)
            # Dịch tax_id sang tên (scientific name)
            names = ncbi.get_taxid_translator(lineage)

            # Duyệt qua tên các tổ tiên để xem có thuộc 4 nhóm chính không
            found = False
            for t_id in lineage:
                name = names.get(t_id, "")
                if name in GROUP_INDICES:
                    group_idx = GROUP_INDICES[name]
                    found = True
                    break

            # Nếu không tìm thấy tên chính xác, thử check rank 'superkingdom'
            if not found:
                ranks = ncbi.get_rank(lineage)
                for t_id, rank in ranks.items():
                    if rank == "superkingdom":
                        sk_name = names.get(t_id, "")
                        if sk_name in GROUP_INDICES:
                            group_idx = GROUP_INDICES[sk_name]
                            break

        except ValueError:
            # TaxID không tồn tại trong DB của NCBI (có thể đã bị xóa hoặc mới)
            pass
        except Exception as e:
            # Các lỗi khác (DB connection, v.v.)
            print(f"Warning: Error mapping TaxID {tax_id}: {e}")
            pass

        vec = [0, 0, 0, 0]
        vec[group_idx] = 1
        tax_map[tax_id] = vec

    return tax_map


# =====================================================================


def get_or_create_embeddings(df_meta, embedding_file_path, desc="Computing Embeddings"):
    """
    1. Identify missing IDs in cache.
    2. Compute embeddings for missing IDs & Update cache (Store only ID & Embedding).
    3. Merge meta with cache.
    4. Generate Superkingdom vector from df_meta['taxonomy'] using ete3 mapping.
    """

    # --- 1. Load Existing Cache (ID only) ---
    existing_ids = set()
    if os.path.exists(embedding_file_path):
        print(f"Reading existing IDs from {embedding_file_path}...")
        try:
            # Chỉ đọc cột id để tiết kiệm RAM
            df_existing_ids = pd.read_parquet(embedding_file_path, columns=["id"])
            existing_ids = set(df_existing_ids["id"].values)
        except Exception as e:
            print(f"Error reading cache index: {e}. Starting fresh.")

    # --- 2. Find Missing IDs ---
    unique_ids_in_meta = df_meta["id"].unique()
    missing_ids = [pid for pid in unique_ids_in_meta if pid not in existing_ids]

    # --- 3. Compute Missing Embeddings ---
    if missing_ids:
        print(f"Found {len(missing_ids)} new proteins to embed.")
        new_entries = []

        # Tạo dict id -> seq để tra cứu nhanh
        meta_unique = df_meta.drop_duplicates(subset=["id"])
        id_to_seq = pd.Series(meta_unique.seq.values, index=meta_unique.id).to_dict()

        for pid in tqdm(missing_ids, desc=desc):
            seq = id_to_seq.get(pid, "")

            emb_vector = None
            try:
                if seq:
                    emb_vector = get_protein_embedding(seq)
            except:
                pass

            if emb_vector is None:
                emb_vector = np.zeros(2560, dtype=np.float32)

            # Chỉ lưu ID và Embedding vào cache
            new_entries.append({"id": pid, "embedding": emb_vector})

        # Append vào file Parquet
        if new_entries:
            df_new = pd.DataFrame(new_entries)
            print(f"Updating cache file {embedding_file_path}...")

            if not os.path.exists(embedding_file_path):
                df_new.to_parquet(embedding_file_path, index=False, engine="pyarrow")
            else:
                import pyarrow.parquet as pq
                import pyarrow as pa

                table_new = pa.Table.from_pandas(df_new)
                table_existing = pq.read_table(embedding_file_path)
                table_combined = pa.concat_tables([table_existing, table_new])
                pq.write_table(table_combined, embedding_file_path)
    else:
        print("All embeddings found in cache.")

    # --- 4. MERGE DATA ---
    print("Merging embeddings into main dataframe...")

    # Load cache (chỉ cột id và embedding)
    df_cache = pd.read_parquet(embedding_file_path, columns=["id", "embedding"])

    # Merge Left: df_meta (có cột taxonomy) + df_cache (có cột embedding)
    df_result = df_meta.merge(df_cache, on="id", how="left")

    # Xử lý embedding null (nếu có)
    if df_result["embedding"].isnull().any():
        print("Warning: Filling null embeddings with zeros.")
        zero_emb = np.zeros(2560, dtype=np.float32)
        null_mask = df_result["embedding"].isnull()
        df_result.loc[null_mask, "embedding"] = df_result.loc[
            null_mask, "embedding"
        ].apply(lambda x: zero_emb)

    # --- 5. MAP TAXONOMY TO VECTOR (ETE3) ---
    print("Generating Superkingdom Vectors using ete3...")

    # Lấy danh sách unique tax_id để map 1 lần (Batch processing)
    # Tối ưu tốc độ thay vì gọi ete3 cho từng dòng
    unique_taxs = df_result["taxonomy"].unique()

    # Build map dict
    tax_map = build_superkingdom_map(unique_taxs)

    # Map vào dataframe
    # Dùng map() của pandas nhanh hơn apply()
    df_result["superkingdom"] = df_result["taxonomy"].map(tax_map)

    # Fallback cho những ID lỗi không có trong map (nếu có)
    # Mặc định về Virus/Other (3)
    default_vec = [0, 0, 0, 1]

    # Check null sau khi map
    if df_result["superkingdom"].isnull().any():
        print("Warning: Some taxonomies could not be mapped. Using default [0,0,0,1].")
        null_tax_mask = df_result["superkingdom"].isnull()
        df_result.loc[null_tax_mask, "superkingdom"] = df_result.loc[
            null_tax_mask, "superkingdom"
        ].apply(lambda x: default_vec)

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

        # Reuse old cache if exists
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
