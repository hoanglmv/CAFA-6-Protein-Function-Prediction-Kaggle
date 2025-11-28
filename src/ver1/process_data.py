import pandas as pd
import re
from Bio import SeqIO
from IPython.display import display
import os

# ==============================
# CẤU HÌNH ĐƯỜNG DẪN
# ==============================

# Lùi lên 2 cấp để về đúng thư mục gốc dự án
# src/ver1 → src → project_root
PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Đường dẫn vào data
BASE_DIR = os.path.join(PROJ_DIR, 'data', 'Train')
TRAIN_FASTA_PATH = os.path.join(BASE_DIR, 'train_sequences.fasta')
TRAIN_TERMS_PATH = os.path.join(BASE_DIR, 'train_terms.tsv')

# ==============================
# BƯỚC 1: ĐỌC FASTA & TRÍCH XUẤT FEATURES
# ==============================
print("1. Đang đọc file FASTA và trích xuất Features (PE, Taxon, Source)...")

ids = []
sequences = []
pe_list = []
taxon_list = []
reviewed_list = []

for record in SeqIO.parse(TRAIN_FASTA_PATH, "fasta"):
    # Xử lý ID
    parts = record.id.split('|')
    clean_id = parts[1] if len(parts) >= 2 else record.id
    
    header = record.description

    pe_match = re.search(r'PE=(\d+)', header)
    pe_val = int(pe_match.group(1)) if pe_match else 0

    ox_match = re.search(r'OX=(\d+)', header)
    ox_val = int(ox_match.group(1)) if ox_match else 0

    is_reviewed = 1 if record.id.startswith("sp|") else 0

    ids.append(clean_id)
    sequences.append(str(record.seq))
    pe_list.append(pe_val)
    taxon_list.append(ox_val)
    reviewed_list.append(is_reviewed)

df_seq = pd.DataFrame({
    'EntryID': ids,
    'sequence': sequences,
    'PE': pe_list,
    'TaxonomyID': taxon_list,
    'Reviewed': reviewed_list
})

print(f"   -> Đã load {len(df_seq)} chuỗi.")
print(f"   -> Ví dụ features dòng đầu: PE={pe_list[0]}, TaxID={taxon_list[0]}")

# ==============================
# BƯỚC 2: ĐỌC FILE TERMS
# ==============================
print("2. Đang đọc file Terms...")
df_terms = pd.read_csv(TRAIN_TERMS_PATH, sep="\t", usecols=['EntryID', 'term'])

# ==============================
# BƯỚC 3: GOM NHÓM LABELS
# ==============================
print("3. Đang gom nhóm các nhãn (Group by)...")
df_labels = df_terms.groupby('EntryID')['term'].apply(list).reset_index()

# ==============================
# BƯỚC 4: GHÉP DỮ LIỆU
# ==============================
print("4. Đang ghép dữ liệu (Inner Join)...")
df_final = pd.merge(df_seq, df_labels, on='EntryID', how='inner')

print("\n=== MẪU DỮ LIỆU SAU KHI GHÉP ===")
if len(df_final) > 0:
    display(df_final.head())

    sample = df_final.iloc[0]
    print(f"[Mẫu 0 - ID {sample['EntryID']}]")
    print(f"- Taxonomy ID: {sample['TaxonomyID']}")
    print(f"- PE Score:    {sample['PE']}")
    print(f"- Reviewed:    {sample['Reviewed']}")
    print(f"- Sequence Len:{len(sample['sequence'])}")
    print(f"- GO Terms:    {sample['term'][:5]}... (Tổng {len(sample['term'])})")
else:
    print("Vẫn rỗng! Kiểm tra lại dữ liệu.")

# ==============================
# LƯU FILE KẾT QUẢ
# ==============================

# Tạo thư mục models/ver1
SAVE_DIR = os.path.join(PROJ_DIR, 'models', 'ver1')
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(SAVE_DIR, 'processed_data.pkl')

print(f"\nĐang lưu dữ liệu vào: {SAVE_PATH} ...")
df_final.to_pickle(SAVE_PATH)

print("Đã lưu thành công! Dùng pd.read_pickle(SAVE_PATH) để load lại.")
