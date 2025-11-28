import pandas as pd
import re
from Bio import SeqIO
from IPython.display import display
import os

# ==============================
# CẤU HÌNH ĐƯỜNG DẪN
# ==============================

# Lùi lên 2 cấp để về đúng thư mục gốc dự án
PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Đường dẫn vào data
BASE_DIR = os.path.join(PROJ_DIR, 'data', 'Train')
TRAIN_FASTA_PATH = os.path.join(BASE_DIR, 'train_sequences.fasta')
TRAIN_TERMS_PATH = os.path.join(BASE_DIR, 'train_terms.tsv')
TRAIN_TAXONOMY_PATH = os.path.join(BASE_DIR, 'train_taxonomy.tsv') # <--- MỚI

# ==============================
# BƯỚC 1: ĐỌC FASTA & TRÍCH XUẤT FEATURES (PE & Reviewed)
# ==============================
print("1. Đang đọc file FASTA và trích xuất Features (PE, Source)...")

ids = []
sequences = []
pe_list = []
reviewed_list = []

# Lưu ý: Chúng ta sẽ không lấy Taxonomy từ Regex nữa mà dùng từ file TSV cho chuẩn
for record in SeqIO.parse(TRAIN_FASTA_PATH, "fasta"):
    # Xử lý ID
    parts = record.id.split('|')
    clean_id = parts[1] if len(parts) >= 2 else record.id
    
    header = record.description

    # Lấy PE (Protein Existence) - Feature quan trọng để đánh trọng số loss
    pe_match = re.search(r'PE=(\d+)', header)
    pe_val = int(pe_match.group(1)) if pe_match else 0

    # Lấy Reviewed (Swiss-Prot vs TrEMBL) - Feature quan trọng về độ tin cậy
    is_reviewed = 1 if record.id.startswith("sp|") else 0

    ids.append(clean_id)
    sequences.append(str(record.seq))
    pe_list.append(pe_val)
    reviewed_list.append(is_reviewed)

df_seq = pd.DataFrame({
    'EntryID': ids,
    'sequence': sequences,
    'PE': pe_list,          # Feature quan trọng 1: Mức độ bằng chứng thực nghiệm
    'Reviewed': reviewed_list # Feature quan trọng 2: Chất lượng nhãn
})

print(f"   -> Đã load {len(df_seq)} chuỗi từ FASTA.")

# ==============================
# ==============================
# BƯỚC 2: ĐỌC FILE TAXONOMY (ĐÃ SỬA LỖI HEADER)
# ==============================
print("2. Đang đọc file Taxonomy (TaxID)...")

# SỬA LỖI: Thêm header=None và tự đặt tên cột
# Vì file không có header nên ta phải chỉ định rõ tên cột
df_tax = pd.read_csv(
    TRAIN_TAXONOMY_PATH, 
    sep="\t", 
    header=None, 
    names=['EntryID', 'taxonomyID']
)

# Chuyển đổi EntryID sang string để khớp với df_seq
df_tax['EntryID'] = df_tax['EntryID'].astype(str)

print(f"   -> Đã load {len(df_tax)} thông tin loài.")
print(f"   -> Kiểm tra 2 dòng đầu:\n{df_tax.head(2)}")
# ==============================
# BƯỚC 3: ĐỌC FILE TERMS
# ==============================
print("3. Đang đọc file Terms...")
df_terms = pd.read_csv(TRAIN_TERMS_PATH, sep="\t", usecols=['EntryID', 'term'])

# ==============================
# BƯỚC 4: GOM NHÓM LABELS
# ==============================
print("4. Đang gom nhóm các nhãn (Group by)...")
df_labels = df_terms.groupby('EntryID')['term'].apply(list).reset_index()

# ==============================
# BƯỚC 5: GHÉP DỮ LIỆU TỔNG HỢP (QUAN TRỌNG)
# ==============================
print("5. Đang ghép toàn bộ dữ liệu (Multiple Merge)...")

# Lần 1: Ghép Sequence + Taxonomy
# Dùng how='left' để giữ lại tất cả sequence, nếu thiếu TaxID thì điền 0 hoặc xử lý sau
df_temp = pd.merge(df_seq, df_tax, on='EntryID', how='left')

# Lần 2: Ghép với Labels (Chỉ lấy những mẫu CÓ nhãn để train)
df_final = pd.merge(df_temp, df_labels, on='EntryID', how='inner')

# Xử lý Missing Values cho Taxonomy (nếu có)
df_final['taxonomyID'] = df_final['taxonomyID'].fillna(0).astype(int)

print("\n=== MẪU DỮ LIỆU SAU KHI GHÉP ===")
if len(df_final) > 0:
    # Sắp xếp lại cột cho đẹp
    cols = ['EntryID', 'taxonomyID', 'PE', 'Reviewed', 'sequence', 'term']
    df_final = df_final[cols]
    
    display(df_final.head())

    sample = df_final.iloc[0]
    print(f"[Kiểm tra mẫu 0 - ID {sample['EntryID']}]")
    print(f"- Taxonomy ID: {sample['taxonomyID']} (Feature Loài - Rất quan trọng)")
    print(f"- PE Score:    {sample['PE']} (Độ tin cậy của protein)")
    print(f"- Reviewed:    {sample['Reviewed']} (Nguồn Swiss-Prot)")
    print(f"- Sequence Len:{len(sample['sequence'])}")
    print(f"- GO Terms:    {sample['term'][:5]}...")
else:
    print("Vẫn rỗng! Kiểm tra lại dữ liệu.")

# ==============================
# LƯU FILE KẾT QUẢ
# ==============================

# Lưu vào thư mục models/ver1
SAVE_DIR = os.path.join(PROJ_DIR, 'models', 'ver3')
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(SAVE_DIR, 'processed_data.pkl')

print(f"\nĐang lưu dữ liệu vào: {SAVE_PATH} ...")
df_final.to_pickle(SAVE_PATH)

print("Đã lưu thành công! Dùng pd.read_pickle(SAVE_PATH) để load lại.")