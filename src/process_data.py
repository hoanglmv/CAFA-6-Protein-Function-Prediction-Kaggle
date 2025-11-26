import pandas as pd
import re  # Thư viện xử lý biểu thức chính quy (Quan trọng để tách PE, OX)
from Bio import SeqIO
from IPython.display import display # Để hiển thị đẹp trong Notebook

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Lưu ý: Nên dùng dấu gạch chéo '/' hoặc r"..." để tránh lỗi đường dẫn trên Windows
BASE_DIR = r'/home/myvh/hoang/CAFA-6-Protein-Function-Prediction-Kaggle/data/Train'
TRAIN_FASTA_PATH = f'{BASE_DIR}/train_sequences.fasta'
TRAIN_TERMS_PATH = f'{BASE_DIR}/train_terms.tsv'

# --- BƯỚC 1: ĐỌC FASTA & TRÍCH XUẤT FEATURES CAO CẤP ---
print("1. Đang đọc file FASTA và trích xuất Features (PE, Taxon, Source)...")

# Tạo các list để chứa dữ liệu
ids = []
sequences = []
pe_list = []        # Protein Existence
taxon_list = []     # Taxonomy ID
reviewed_list = []  # 1 = Swiss-Prot (Tốt), 0 = TrEMBL

# Duyệt qua từng dòng trong file FASTA
for record in SeqIO.parse(TRAIN_FASTA_PATH, "fasta"):
    # 1. Xử lý ID (Lấy phần giữa dấu |)
    parts = record.id.split('|')
    if len(parts) >= 2:
        clean_id = parts[1]
    else:
        clean_id = record.id
    
    # 2. Trích xuất thông tin từ Header (Description)
    header = record.description
    
    # Lấy PE (Protein Existence): Tìm chuỗi "PE=số"
    pe_match = re.search(r'PE=(\d+)', header)
    pe_val = int(pe_match.group(1)) if pe_match else 0
    
    # Lấy Taxonomy ID (OX): Tìm chuỗi "OX=số"
    ox_match = re.search(r'OX=(\d+)', header)
    ox_val = int(ox_match.group(1)) if ox_match else 0
    
    # Xác định nguồn (Reviewed hay chưa): Nếu bắt đầu bằng "sp|" là Reviewed
    is_reviewed = 1 if record.id.startswith("sp|") else 0
    
    # Lưu vào list
    ids.append(clean_id)
    sequences.append(str(record.seq))
    pe_list.append(pe_val)
    taxon_list.append(ox_val)
    reviewed_list.append(is_reviewed)

# Tạo DataFrame với đầy đủ features
df_seq = pd.DataFrame({
    'EntryID': ids,
    'sequence': sequences,
    'PE': pe_list,          # Mức độ bằng chứng (1 là tốt nhất)
    'TaxonomyID': taxon_list, # Mã loài
    'Reviewed': reviewed_list # 1 là dữ liệu chất lượng cao
})

print(f"   -> Đã load {len(df_seq)} chuỗi.")
print(f"   -> Ví dụ features dòng đầu: PE={pe_list[0]}, TaxID={taxon_list[0]}")

# --- BƯỚC 2: ĐỌC FILE TERMS ---
print("2. Đang đọc file Terms...")
df_terms = pd.read_csv(TRAIN_TERMS_PATH, sep="\t", usecols=['EntryID', 'term'])

# --- BƯỚC 3: GOM NHÓM LABELS ---
print("3. Đang gom nhóm các nhãn (Group by)...")
df_labels = df_terms.groupby('EntryID')['term'].apply(list).reset_index()

# --- BƯỚC 4: GHÉP DỮ LIỆU (MERGE) ---
print("4. Đang ghép dữ liệu (Inner Join)...")
df_final = pd.merge(df_seq, df_labels, on='EntryID', how='inner')

# --- KẾT QUẢ & HIỂN THỊ ---
print("\n=== MẪU DỮ LIỆU SAU KHI GHÉP (ĐẦY ĐỦ FEATURES) ===")
if len(df_final) > 0:
    # Hiển thị bảng đẹp
    display(df_final.head())
    
    # Kiểm tra chi tiết mẫu đầu tiên
    sample = df_final.iloc[0]
    print(f"\n[Kiểm tra chi tiết mẫu số 0 - ID: {sample['EntryID']}]")
    print(f"- Taxonomy ID: {sample['TaxonomyID']} (Loài)")
    print(f"- PE Score:    {sample['PE']} (1=Evidence, cao nhất)")
    print(f"- Reviewed:    {sample['Reviewed']} (1=Swiss-Prot)")
    print(f"- Sequence Len:{len(sample['sequence'])}")
    print(f"- GO Terms:    {sample['term'][:5]}... (Tổng {len(sample['term'])} terms)")
else:
    print("Vẫn rỗng! Hãy kiểm tra lại ID.")

import os

# Đường dẫn lưu file (Cùng thư mục data)
SAVE_PATH = r'/home/myvh/hoang/CAFA-6-Protein-Function-Prediction-Kaggle/data/processed_data.pkl'

print(f"Đang lưu dữ liệu vào: {SAVE_PATH} ...")
# Dùng pickle để lưu giữ nguyên định dạng List và các cột
df_final.to_pickle(SAVE_PATH)

print("Đã lưu thành công! Lần sau bạn chỉ cần pd.read_pickle(SAVE_PATH) là xong.")