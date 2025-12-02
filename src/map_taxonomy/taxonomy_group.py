import os
import pickle
import pandas as pd
import numpy as np
from Bio import SeqIO
from ete3 import NCBITaxa
from tqdm import tqdm

# ==========================================
# CẤU HÌNH
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TRAIN_TAX_PATH = os.path.join(BASE_DIR, "Train", "train_taxonomy.tsv")
TEST_FASTA_PATH = os.path.join(BASE_DIR, "Test", "testsuperset.fasta")
OUTPUT_PATH = os.path.join("models", "ver5", "taxonomy_group_mapping.pkl") # Nơi lưu file map

# Tạo thư mục output nếu chưa có
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ==========================================
# CLASS XỬ LÝ NHÓM (GROUPER)
# ==========================================
class TaxonGrouper:
    def __init__(self):
        print("⏳ Khởi tạo NCBI Database (Lần đầu chạy sẽ tải ~300MB)...")
        self.ncbi = NCBITaxa() # Tự động tải/update database nếu cần
        
        # Định nghĩa 4 nhóm chính (Superkingdoms) theo NCBI TaxID
        self.SUPERKINGDOMS = {
            2759: 0,   # Eukaryota (Nhân thực: Người, Nấm, Cây...)
            2: 1,      # Bacteria (Vi khuẩn)
            2157: 2,   # Archaea (Cổ khuẩn)
            10239: 3   # Viruses (Virus)
        }
        self.num_classes = 4
        self.cache = {} # Lưu kết quả tra cứu để tăng tốc

    def get_one_hot(self, tax_id):
        # 1. Kiểm tra cache
        if tax_id in self.cache:
            return self.cache[tax_id]
        
        # Vector mặc định [0,0,0,0] (Unknown)
        vector = np.zeros(self.num_classes, dtype=np.float32)
        
        try:
            tid = int(tax_id)
            # 2. Lấy dòng dõi tổ tiên (Lineage)
            # get_lineage trả về list các ID từ gốc đến ngọn
            lineage = self.ncbi.get_lineage(tid)
            
            # 3. Kiểm tra xem tổ tiên có nằm trong 4 nhóm chính không
            for ancestor in lineage:
                if ancestor in self.SUPERKINGDOMS:
                    idx = self.SUPERKINGDOMS[ancestor]
                    vector[idx] = 1.0
                    break 
        except Exception:
            # Lỗi thường gặp: TaxID không có trong DB của ete3 (mới quá hoặc bị xóa)
            # Giữ nguyên vector 0
            pass
            
        # 4. Lưu cache
        self.cache[tax_id] = vector
        return vector

# ==========================================
# MAIN PROCESS
# ==========================================
def main():
    grouper = TaxonGrouper()
    final_mapping = {} # Dictionary: {EntryID: OneHotVector}
    
    # --- BƯỚC 1: XỬ LÝ TẬP TRAIN ---
    print(f"📖 Đang đọc Train: {TRAIN_TAX_PATH}")
    # File tsv không header: Cột 0 là EntryID, Cột 1 là TaxID
    try:
        df_train = pd.read_csv(TRAIN_TAX_PATH, sep='\t', header=None, names=['EntryID', 'TaxID'], dtype=str)
        unique_train_tax = df_train['TaxID'].unique()
        print(f"   -> Tìm thấy {len(unique_train_tax)} loài trong Train.")
        
        # Pre-calculate cho các loài (để đỡ gọi ete3 nhiều lần)
        print("   -> Đang nhóm các loài Train...")
        tax_to_vec_train = {}
        for tax in tqdm(unique_train_tax):
            tax_to_vec_train[tax] = grouper.get_one_hot(tax)
            
        # Map vào từng EntryID
        print("   -> Mapping EntryID...")
        for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
            final_mapping[row['EntryID']] = tax_to_vec_train[row['TaxID']]
            
    except Exception as e:
        print(f"❌ Lỗi đọc Train: {e}")

    # --- BƯỚC 2: XỬ LÝ TẬP TEST ---
    print(f"📖 Đang đọc Test: {TEST_FASTA_PATH}")
    test_entries = []
    unique_test_tax = set()
    
    # Parse FASTA để lấy ID và TaxID
    for record in tqdm(SeqIO.parse(TEST_FASTA_PATH, "fasta")):
        # Header: "A0A0C5B5G6 9606" -> ID: A0A0C5B5G6, Tax: 9606
        parts = record.description.split()
        entry_id = parts[0]
        tax_id = parts[1] if len(parts) >= 2 and parts[1].isdigit() else "0"
        
        test_entries.append((entry_id, tax_id))
        unique_test_tax.add(tax_id)
        
    print(f"   -> Tìm thấy {len(unique_test_tax)} loài trong Test.")
    
    # Pre-calculate cho Test
    print("   -> Đang nhóm các loài Test...")
    tax_to_vec_test = {}
    for tax in tqdm(unique_test_tax):
        tax_to_vec_test[tax] = grouper.get_one_hot(tax)
        
    # Map vào EntryID
    print("   -> Mapping Test Entries...")
    for entry_id, tax_id in test_entries:
        final_mapping[entry_id] = tax_to_vec_test[tax_id]

    # --- BƯỚC 3: LƯU KẾT QUẢ ---
    print(f"💾 Đang lưu kết quả vào {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(final_mapping, f)
        
    print("✅ HOÀN TẤT!")
    print(f"- Tổng số Protein đã map: {len(final_mapping)}")
    
    # Test thử 1 mẫu
    sample_id = list(final_mapping.keys())[0]
    print(f"- Mẫu ({sample_id}): {final_mapping[sample_id]}")

if __name__ == "__main__":
    main()