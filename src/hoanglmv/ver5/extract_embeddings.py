import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO

# ==========================================
# CẤU HÌNH
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'esm2_ver1')

# Input Files
TRAIN_PKL = os.path.join(BASE_DIR, 'models', 'ver5', 'processed_data.pkl') # Lấy từ bước process cũ
TEST_FASTA = os.path.join(DATA_DIR, 'Test', 'testsuperset.fasta')

# Output Files
TRAIN_EMB_PATH = os.path.join(MODEL_DIR, 'train_embeddings.npy')
TEST_EMB_PATH = os.path.join(MODEL_DIR, 'test_embeddings.npy')
TEST_IDS_PATH = os.path.join(MODEL_DIR, 'test_ids.pkl') # Lưu thứ tự ID test để khớp

# Model Config
# Bạn có thể đổi sang 'facebook/esm2_t12_35M_UR50D' (mạnh hơn) nếu GPU đủ VRAM
MODEL_NAME = "facebook/esm2_t6_8M_UR50D" 
BATCH_SIZE = 32 # Tăng giảm tùy VRAM
MAX_LEN = 512

os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Running on {DEVICE} with model {MODEL_NAME}")

# ==========================================
# HÀM TRÍCH XUẤT
# ==========================================
def extract_embeddings(sequence_list, model_name, save_path):
    print(f"🚀 Loading Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(sequence_list), BATCH_SIZE)):
        batch_seqs = sequence_list[i : i + BATCH_SIZE]
        
        # Tokenize
        inputs = tokenizer(
            batch_seqs, 
            padding=True, 
            truncation=True, 
            max_length=MAX_LEN, 
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Lấy Mean Pooling của last_hidden_state (bỏ qua padding mask)
            # last_hidden_state: (Batch, Seq_Len, Dim)
            
            last_hidden = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            
            sum_embeddings = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            
            mean_embeddings = sum_embeddings / sum_mask
            embeddings.append(mean_embeddings.cpu().numpy())
            
    # Concatenate and Save
    full_embeddings = np.vstack(embeddings)
    print(f"💾 Saving to {save_path} | Shape: {full_embeddings.shape}")
    np.save(save_path, full_embeddings)
    return full_embeddings

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- 1. EXTRACT TRAIN ---
    if os.path.exists(TRAIN_EMB_PATH):
        print("✅ Train embeddings already exist. Skipping.")
    else:
        print("📖 Reading Train Data...")
        df_train = pd.read_pickle(TRAIN_PKL)
        sequences = df_train['sequence'].tolist()
        # Thay thế các axit amin lạ (U, Z, O, B) bằng X (Unknown) cho ESM
        sequences = [s.replace('U','X').replace('Z','X').replace('O','X').replace('B','X') for s in sequences]
        
        extract_embeddings(sequences, MODEL_NAME, TRAIN_EMB_PATH)

    # --- 2. EXTRACT TEST ---
    if os.path.exists(TEST_EMB_PATH):
        print("✅ Test embeddings already exist. Skipping.")
    else:
        print("📖 Reading Test FASTA...")
        test_ids = []
        test_seqs = []
        for record in SeqIO.parse(TEST_FASTA, "fasta"):
            test_ids.append(record.id)
            seq = str(record.seq).replace('U','X').replace('Z','X').replace('O','X').replace('B','X')
            test_seqs.append(seq)
            
        # Lưu lại danh sách ID để dùng khi tạo submission
        with open(TEST_IDS_PATH, 'wb') as f:
            pickle.dump(test_ids, f)
            
        extract_embeddings(test_seqs, MODEL_NAME, TEST_EMB_PATH)
        
    print("\n🎉 Xong phần trích xuất đặc trưng!")