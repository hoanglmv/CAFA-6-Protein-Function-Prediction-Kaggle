import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CẤU HÌNH
# ==========================================
BASE_DIR = "data"
# File chứa danh sách ID loài trong tập Train (Nguồn tham chiếu chuẩn)
TRAIN_TAX_PATH = os.path.join(BASE_DIR, "Train", "train_taxonomy.tsv")

# File chứa tên loài (ID -> Name) do Kaggle cung cấp
# File này cần có cột TaxID và ScientificName
TAXON_LIST_PATH = os.path.join(BASE_DIR, "Test", "testsuperset-taxon-list.tsv")

# File Output
OUTPUT_FILE = "taxonomy_mapping.tsv"

# Model Config
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"

# ==========================================
# 2. XỬ LÝ DỮ LIỆU
# ==========================================
def main():
    print(f"🚀 Bắt đầu Mapping với Model: {MODEL_NAME}")

    # --- BƯỚC 1: LẤY DANH SÁCH KNOWN ID TỪ TRAIN ---
    print(f"📖 Đang đọc ID từ Train: {TRAIN_TAX_PATH}")
    try:
        # File train_taxonomy.tsv không có header, cột 2 (index 1) là TaxID
        df_train_tax = pd.read_csv(TRAIN_TAX_PATH, sep='\t', header=None, dtype=str)
        known_tax_ids = set(df_train_tax.iloc[:, 1].unique()) 
    except Exception as e:
        print(f"⚠️ Lỗi đọc file Train Tax: {e}")
        return

    print(f"✅ Tìm thấy {len(known_tax_ids)} loài đã biết (Known) trong tập Train.")

    # --- BƯỚC 2: ĐỌC DANH SÁCH TÊN LOÀI (SUPERSET) ---
    print(f"📖 Đang đọc danh sách tên loài: {TAXON_LIST_PATH}")
    try:
        # Cố gắng đọc file taxon list
        # Kiểm tra xem file có header hay không, nếu không thì gán tên cột
        df_taxons = pd.read_csv(TAXON_LIST_PATH, sep='\t', encoding='latin-1', dtype=str)
        
        # Logic đoán header: nếu cột đầu không phải tên là 'TaxID' hoặc tương tự
        if 'Taxon_ID' not in df_taxons.columns and 'TaxID' not in df_taxons.columns:
             # Load lại với header=None
             df_taxons = pd.read_csv(TAXON_LIST_PATH, sep='\t', header=None, names=['TaxID', 'SpeciesName'], encoding='latin-1', dtype=str)
        else:
             # Chuẩn hóa tên cột về 'TaxID' và 'SpeciesName'
             col_map = {df_taxons.columns[0]: 'TaxID', df_taxons.columns[1]: 'SpeciesName'}
             df_taxons.rename(columns=col_map, inplace=True)
             
    except Exception as e:
        print(f"❌ Lỗi đọc file Taxon List: {e}")
        return

    # --- BƯỚC 3: PHÂN LOẠI KNOWN / UNKNOWN ---
    final_mapping = []
    
    # Nhóm A: Known (Đích đến để tra cứu)
    df_known_source = df_taxons[df_taxons['TaxID'].isin(known_tax_ids)].copy()
    
    # Nhóm B: Unknown (Cần map)
    df_unknown_target = df_taxons[~df_taxons['TaxID'].isin(known_tax_ids)].copy()
    
    print(f"📊 Thống kê dữ liệu:")
    print(f"   - Known Species (Source): {len(df_known_source)}")
    print(f"   - Unknown Species (Target): {len(df_unknown_target)}")

    # === PHẦN 1: DIRECT MATCH (ƯU TIÊN TUYỆT ĐỐI) ===
    print("\n🔹 [1/2] Đang xử lý Direct Match (Khớp chính xác)...")
    # Với những loài đã có trong train, map chính nó sang chính nó
    for _, row in tqdm(df_known_source.iterrows(), total=len(df_known_source)):
        final_mapping.append({
            'Original_TaxID': row['TaxID'],
            'Mapped_TaxID': row['TaxID'], 
            'Score': 1.0,
            'Method': 'Direct Match'
        })

    # === PHẦN 2: SEMANTIC MATCH VỚI QWEN ===
    if len(df_unknown_target) > 0:
        print(f"\n🔹 [2/2] Đang xử lý Semantic Match với {MODEL_NAME}...")
        
        # Load Model Qwen
        # trust_remote_code=True là bắt buộc với model Qwen
        print("   -> Loading Model (có thể mất vài phút)...")
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        
        # Dữ liệu nguồn (Known)
        known_names = df_known_source['SpeciesName'].tolist()
        known_ids = df_known_source['TaxID'].tolist()
        
        # Dữ liệu đích (Unknown)
        unknown_names = df_unknown_target['SpeciesName'].tolist()
        unknown_ids = df_unknown_target['TaxID'].tolist()
        
        if len(known_names) == 0:
            print("⚠️ Cảnh báo: Tập Train rỗng! Không có đích để map.")
        else:
            print(f"   -> Encoding {len(known_names)} Known Species...")
            # convert_to_tensor=True để tính similarity bằng GPU cho nhanh
            known_embeddings = model.encode(known_names, convert_to_tensor=True, show_progress_bar=True)

            print(f"   -> Encoding {len(unknown_names)} Unknown Species...")
            unknown_embeddings = model.encode(unknown_names, convert_to_tensor=True, show_progress_bar=True)

            print("   -> Calculating Similarity & Mapping...")
            # Sử dụng hàm similarity có sẵn của SentenceTransformer (trả về Tensor)
            # Matrix kích thước: (Num_Unknown, Num_Known)
            similarities = model.similarity(unknown_embeddings, known_embeddings)
            
            # Chuyển về CPU/Numpy để xử lý vòng lặp
            similarities = similarities.cpu().numpy()
            
            for i in tqdm(range(len(unknown_ids)), desc="Mapping"):
                # Tìm index của loài Known có điểm cao nhất
                best_idx = np.argmax(similarities[i])
                best_score = similarities[i][best_idx]
                
                mapped_id = known_ids[best_idx]
                mapped_name = known_names[best_idx]
                original_name = unknown_names[i]
                
                final_mapping.append({
                    'Original_TaxID': unknown_ids[i],
                    'Mapped_TaxID': mapped_id,
                    'Score': f"{best_score:.4f}",
                    'Method': f"Semantic: {original_name} -> {mapped_name}"
                })

    # --- BƯỚC 4: LƯU FILE ---
    print(f"\n💾 Đang lưu kết quả: {OUTPUT_FILE}")
    df_result = pd.DataFrame(final_mapping)
    
    # Sắp xếp: Direct Match lên đầu
    df_result.sort_values(by=['Score'], ascending=False, inplace=True)
    
    df_result.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print("✅ Hoàn tất!")
    print(df_result.head())

if __name__ == "__main__":
    # Kiểm tra GPU
    if torch.cuda.is_available():
        print(f"⚡ Đang chạy trên GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Đang chạy trên CPU (Sẽ chậm với model 4B params)")
        
    main()