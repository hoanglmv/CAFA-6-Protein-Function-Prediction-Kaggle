import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')) 
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 1. File chứa ID loài đã học (Train)
TRAIN_TAX_PATH = os.path.join(DATA_DIR, "Train", "train_taxonomy.tsv")

# 2. File chứa Tên loài (Của cả Train và Test)
# File này Kaggle cung cấp: TaxID <tab> Scientific Name
TAXON_LIST_PATH = os.path.join(DATA_DIR, "Test", "testsuperset-taxon-list.tsv")

OUTPUT_FILE = os.path.join(BASE_DIR, "models", "ver5", "taxonomy_mapping.tsv")

def main():
    print("🚀 Bắt đầu Mapping Taxonomy...")

    # --- 1. Load Known IDs ---
    print(f"📖 Đọc Train IDs: {TRAIN_TAX_PATH}")
    # Train tax file: EntryID <tab> TaxID
    df_train = pd.read_csv(TRAIN_TAX_PATH, sep='\t', header=None, dtype=str)
    # Lấy danh sách các TaxID duy nhất có trong tập Train
    known_tax_ids = set(df_train.iloc[:, 1].unique())
    print(f"✅ Train set có {len(known_tax_ids)} loài.")

    # --- 2. Load Names ---
    print(f"📖 Đọc danh sách tên loài: {TAXON_LIST_PATH}")
    # File này có thể có header hoặc không, check kỹ
    try:
        df_taxons = pd.read_csv(TAXON_LIST_PATH, sep='\t', encoding='latin-1', dtype=str)
        # Chuẩn hóa tên cột
        if 'Taxon_ID' not in df_taxons.columns and len(df_taxons.columns) >= 2:
             df_taxons = pd.read_csv(TAXON_LIST_PATH, sep='\t', header=None, names=['TaxID', 'SpeciesName'], encoding='latin-1', dtype=str)
        else:
             # Rename cột đầu thành TaxID, cột 2 thành SpeciesName
             df_taxons.rename(columns={df_taxons.columns[0]: 'TaxID', df_taxons.columns[1]: 'SpeciesName'}, inplace=True)
    except Exception as e:
        print(f"❌ Lỗi đọc file Taxon List: {e}")
        return

    # --- 3. Phân loại Known vs Unknown ---
    # Known: Những loài có trong Train (Đích đến)
    df_known = df_taxons[df_taxons['TaxID'].isin(known_tax_ids)].copy()
    
    # Unknown: Những loài chỉ có trong Test (Cần map)
    df_unknown = df_taxons[~df_taxons['TaxID'].isin(known_tax_ids)].copy()
    
    print(f"📊 Thống kê:\n - Known Species: {len(df_known)}\n - Unknown Species: {len(df_unknown)}")

    final_mapping = []

    # --- 4. Direct Match (Map chính nó) ---
    print("🔹 [1/2] Mapping Direct...")
    for _, row in df_known.iterrows():
        final_mapping.append({
            'Original_ID': row['TaxID'],
            'Mapped_ID': row['TaxID'],
            'Score': 1.0,
            'Method': 'Direct'
        })

    # --- 5. Semantic Match (Tìm loài gần nhất) ---
    if len(df_unknown) > 0:
        print("🔹 [2/2] Mapping Semantic (Embedding)...")
        # Load model siêu nhẹ chuyên trị so sánh câu
        model = SentenceTransformer('all-MiniLM-L6-v2') 
        
        known_names = df_known['SpeciesName'].tolist()
        known_ids = df_known['TaxID'].tolist()
        
        unknown_names = df_unknown['SpeciesName'].tolist()
        unknown_ids = df_unknown['TaxID'].tolist()
        
        print("   -> Embedding Known Names...")
        emb_known = model.encode(known_names, show_progress_bar=True)
        
        print("   -> Embedding Unknown Names...")
        emb_unknown = model.encode(unknown_names, show_progress_bar=True)
        
        print("   -> Calculating Similarity...")
        sim_matrix = cosine_similarity(emb_unknown, emb_known)
        
        for i in tqdm(range(len(unknown_ids))):
            best_idx = np.argmax(sim_matrix[i])
            best_score = sim_matrix[i][best_idx]
            
            final_mapping.append({
                'Original_ID': unknown_ids[i],
                'Mapped_ID': known_ids[best_idx], # ID loài Known giống nhất
                'Score': best_score,
                'Method': 'Semantic'
            })

    # --- 6. Save ---
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    pd.DataFrame(final_mapping).to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"✅ Xong! File map lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()