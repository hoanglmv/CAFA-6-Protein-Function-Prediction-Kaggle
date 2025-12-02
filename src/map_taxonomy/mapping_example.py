import os
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (Sửa lại cho đúng máy bạn)
# ==============================================================================
BASE_DIR = r"D:\vhproj\CAFA-6-Protein-Function-Prediction-Kaggle"
TRAIN_TAX_PATH  = os.path.join(BASE_DIR, "data", "Train", "train_taxonomy.tsv")
TEST_FASTA_PATH = os.path.join(BASE_DIR, "data", "Test", "testsuperset.fasta")
OUTPUT_FILE     = os.path.join(BASE_DIR, "mapping.tsv")

# ==============================================================================
# 2. XỬ LÝ DỮ LIỆU TRAIN (Tạo từ điển tra cứu)
# ==============================================================================
def load_train_species_map(tsv_path):
    print(f"📖 Đang đọc Train Taxonomy: {tsv_path}")
    
    # Đọc file TSV (Giả định không có header)
    # Cột 0: EntryID, Cột 1: TaxID
    try:
        df = pd.read_csv(tsv_path, sep='\t', header=None, names=['EntryID', 'TaxID'], dtype=str)
    except Exception as e:
        print(f"❌ Lỗi đọc file Train: {e}")
        return {}

    # Tạo map: {TaxID -> Representative_EntryID}
    # Mỗi loài chỉ cần lấy 1 ID đại diện để map ngược lại
    df_unique = df.drop_duplicates(subset=['TaxID'], keep='first')
    tax_map = dict(zip(df_unique['TaxID'], df_unique['EntryID']))
    
    print(f"✅ Đã lập chỉ mục cho {len(tax_map)} loài trong tập Train.")
    return tax_map

# ==============================================================================
# 3. XỬ LÝ DỮ LIỆU TEST & MAPPING (SỬA ĐỔI LOGIC)
# ==============================================================================
def create_mapping_file(test_fasta, train_map, output_path):
    print(f"📖 Đang quét Test FASTA: {test_fasta}")
    
    results = []
    found_count = 0
    not_found_count = 0
    
    for record in tqdm(SeqIO.parse(test_fasta, "fasta")):
        # Header thực tế: "A0A0C5B5G6 9606"
        # Biopython đã tách dấu > ra rồi
        description = record.description
        parts = description.split()
        
        test_id = parts[0] # "A0A0C5B5G6"
        
        # Lấy TaxID (thường là phần tử thứ 2)
        if len(parts) >= 2 and parts[1].isdigit():
            tax_id = parts[1] # "9606"
        else:
            tax_id = None
            
        # Mapping
        mapped_train_id = "None"
        if tax_id and tax_id in train_map:
            mapped_train_id = train_map[tax_id]
            found_count += 1
        else:
            not_found_count += 1
            
        results.append({
            'Train_ID_Mapped': mapped_train_id,
            'Test_ID': test_id
        })
        
    # ==========================================================================
    # 4. XUẤT FILE
    # ==========================================================================
    print(f"💾 Đang ghi file kết quả: {output_path}")
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, sep='\t', index=False)
    
    print("="*40)
    print(f"✅ HOÀN TẤT!")
    print(f"- Tổng số mẫu Test: {len(results)}")
    print(f"- Map thành công (Tìm thấy loài): {found_count}")
    print(f"- Map thất bại (Loài lạ/Lỗi): {not_found_count}")
    print(f"- File lưu tại: {output_path}")
    print("="*40)
    
    # In thử vài dòng đầu để kiểm tra
    print(df_out.head())

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(TRAIN_TAX_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {TRAIN_TAX_PATH}")
    elif not os.path.exists(TEST_FASTA_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {TEST_FASTA_PATH}")
    else:
        # 1. Load Train
        train_map = load_train_species_map(TRAIN_TAX_PATH)
        
        # 2. Process
        if train_map:
            create_mapping_file(TEST_FASTA_PATH, train_map, OUTPUT_FILE)