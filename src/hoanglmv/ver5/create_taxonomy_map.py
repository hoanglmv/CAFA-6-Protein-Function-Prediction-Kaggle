import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ete3 import NCBITaxa

# ==========================================
# CẤU HÌNH
# ==========================================
# Lùi 3 cấp: src/hoanglmv/ver5 -> hoanglmv -> src -> PROJ_DIR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

TRAIN_TAX_PATH = os.path.join(DATA_DIR, "Train", "train_taxonomy.tsv")
TAXON_LIST_PATH = os.path.join(DATA_DIR, "Test", "testsuperset-taxon-list.tsv")
OUTPUT_FILE = os.path.join(BASE_DIR, "models", "ver5", "taxonomy_mapping.tsv")

def get_train_taxids():
    print(f"📖 Đọc Train Taxonomy: {TRAIN_TAX_PATH}")
    try:
        df = pd.read_csv(TRAIN_TAX_PATH, sep='\t', header=None, dtype=str)
        # Giả sử cột 1 là TaxID
        tax_ids = set(df.iloc[:, 1].unique().astype(int))
        return tax_ids
    except Exception as e:
        print(f"❌ Lỗi đọc file Train: {e}")
        return set()

def get_test_taxons():
    print(f"📖 Đọc Test Taxonomy List: {TAXON_LIST_PATH}")
    try:
        # SỬA LỖI QUAN TRỌNG: Xử lý Header và dòng rác
        df = pd.read_csv(TAXON_LIST_PATH, sep='\t', encoding='latin-1', dtype=str, on_bad_lines='skip')
        
        # Đổi tên cột chuẩn
        if 'Taxon_ID' in df.columns: df.rename(columns={'Taxon_ID': 'TaxID'}, inplace=True)
        elif 'ID' in df.columns: df.rename(columns={'ID': 'TaxID'}, inplace=True)
        elif len(df.columns) >= 2 and 'TaxID' not in df.columns:
             df = pd.read_csv(TAXON_LIST_PATH, sep='\t', header=None, names=['TaxID', 'Name'], encoding='latin-1', dtype=str)

        # Lọc bỏ các dòng mà TaxID không phải là số (VD: dòng header lặp lại)
        df = df[pd.to_numeric(df['TaxID'], errors='coerce').notnull()]
        
        # Ép kiểu an toàn
        df['TaxID'] = df['TaxID'].astype(int)
        return df
    except Exception as e:
        print(f"❌ Lỗi đọc file Test Taxon: {e}")
        return pd.DataFrame()

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print("⏳ Khởi tạo NCBI DB...")
    ncbi = NCBITaxa() # Cần internet lần đầu
    
    train_tax_ids = get_train_taxids()
    df_test = get_test_taxons()
    
    if len(train_tax_ids) == 0 or df_test.empty: return

    # Validate Train IDs trong NCBI
    valid_train_set = set()
    for tid in train_tax_ids:
        try:
            ncbi.get_rank([tid]); valid_train_set.add(tid)
        except: continue
    
    print(f"✅ Train Set Valid: {len(valid_train_set)} loài.")
    
    mapping_results = []
    test_tax_ids = df_test['TaxID'].unique()
    
    print("🚀 Bắt đầu Mapping (Phylogenetic)...")
    for test_id in tqdm(test_tax_ids):
        # 1. Direct Match
        if test_id in valid_train_set:
            mapping_results.append({'Original_ID': test_id, 'Mapped_ID': test_id, 'Method': 'Direct'})
            continue
            
        # 2. Phylogenetic Search
        try:
            lineage = ncbi.get_lineage(test_id)[::-1] # Từ loài ngược lên gốc
            found = False
            for ancestor in lineage:
                if ancestor == test_id: continue
                # Tìm con cháu của tổ tiên này có mặt trong tập Train
                descendants = ncbi.get_descendant_taxa(ancestor, collapse_subspecies=True)
                relatives = valid_train_set.intersection(set(descendants))
                if relatives:
                    best_match = list(relatives)[0] # Lấy đại diện đầu tiên
                    mapping_results.append({'Original_ID': test_id, 'Mapped_ID': best_match, 'Method': 'Phylogenetic'})
                    found = True
                    break
            
            if not found:
                mapping_results.append({'Original_ID': test_id, 'Mapped_ID': 9606, 'Method': 'Fallback'}) # Mặc định người
        except:
            mapping_results.append({'Original_ID': test_id, 'Mapped_ID': 9606, 'Method': 'Error'})

    # Save
    pd.DataFrame(mapping_results).to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"✅ Xong! File map tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()