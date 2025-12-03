import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import pickle
import sys

# ==========================================
# CẤU HÌNH
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'esm2_ver1')

# Input Files
TRAIN_PKL = os.path.join(BASE_DIR, 'models', 'ver5', 'processed_data.pkl') 
TEST_FASTA = os.path.join(DATA_DIR, 'Test', 'testsuperset.fasta')

# Output Files
TRAIN_EMB_PATH = os.path.join(MODEL_DIR, 'train_embeddings.npy')
TEST_EMB_PATH = os.path.join(MODEL_DIR, 'test_embeddings.npy')
TEST_IDS_PATH = os.path.join(MODEL_DIR, 'test_ids.pkl') 

# Model Config
# RTX 3060 12GB dư sức chạy bản 650M (t33). Hãy dùng bản này để có kết quả tốt hơn bản 8M (t6).
# MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # Quá yếu
MODEL_NAME = "facebook/esm2_t33_650M_UR50D" # Khuyên dùng (Vector 1280 chiều)

BATCH_SIZE = 32 # Nếu bị tràn VRAM (OOM), hãy giảm xuống 16
MAX_LEN = 512

os.makedirs(MODEL_DIR, exist_ok=True)

# --- CẤU HÌNH GPU CHẶT CHẼ ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ ĐÃ KÍCH HOẠT GPU: {gpu_name}")
    # Bật chế độ tối ưu toán học cho GPU (Optional)
    torch.backends.cudnn.benchmark = True
else:
    print("❌ LỖI NGHIÊM TRỌNG: Không tìm thấy GPU NVIDIA!")
    print("   Code sẽ chạy trên CPU và mất hàng giờ đồng hồ.")
    print("   👉 Hãy kiểm tra lại cài đặt PyTorch với lệnh: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    # Dừng chương trình ngay lập tức để bạn không tốn thời gian chạy CPU
    sys.exit("Dừng chương trình do không có GPU.")

print(f"⚙️ Model: {MODEL_NAME}")

# ==========================================
# HÀM TRÍCH XUẤT
# ==========================================
def extract_embeddings(sequence_list, model_name, save_path):
    print(f"🚀 Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    
    # Sử dụng Mixed Precision (FP16) để chạy nhanh hơn và tiết kiệm VRAM trên RTX 3060
    # scaler = torch.amp.GradScaler() # Chỉ dùng khi training, inference dùng autocast là đủ
    
    embeddings = []
    
    print(f"🔄 Đang xử lý {len(sequence_list)} chuỗi...")
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
            # Chạy chế độ Mixed Precision (tự động dùng FP16)
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state
                
                # Mean Pooling (có tính đến Attention Mask)
                mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                
            # Chuyển về CPU numpy để tiết kiệm VRAM GPU
            embeddings.append(mean_embeddings.float().cpu().numpy())
            
    # Gộp và Lưu
    full_embeddings = np.vstack(embeddings)
    print(f"💾 Đang lưu file npy: {save_path} | Shape: {full_embeddings.shape}")
    np.save(save_path, full_embeddings)
    return full_embeddings

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- 1. EXTRACT TRAIN ---
    if os.path.exists(TRAIN_EMB_PATH):
        print(f"✅ File {TRAIN_EMB_PATH} đã tồn tại. Bỏ qua.")
    else:
        print("📖 Đọc dữ liệu Train...")
        if not os.path.exists(TRAIN_PKL):
            print(f"❌ Không tìm thấy file {TRAIN_PKL}")
            sys.exit()
            
        df_train = pd.read_pickle(TRAIN_PKL)
        sequences = df_train['sequence'].tolist()
        # Vệ sinh chuỗi protein
        sequences = [s.replace('U','X').replace('Z','X').replace('O','X').replace('B','X') for s in sequences]
        
        extract_embeddings(sequences, MODEL_NAME, TRAIN_EMB_PATH)

    # --- 2. EXTRACT TEST ---
    if os.path.exists(TEST_EMB_PATH):
        print(f"✅ File {TEST_EMB_PATH} đã tồn tại. Bỏ qua.")
    else:
        print("📖 Đọc dữ liệu Test...")
        if not os.path.exists(TEST_FASTA):
            print(f"❌ Không tìm thấy file {TEST_FASTA}")
            sys.exit()

        test_ids = []
        test_seqs = []
        for record in SeqIO.parse(TEST_FASTA, "fasta"):
            test_ids.append(record.id)
            seq = str(record.seq).replace('U','X').replace('Z','X').replace('O','X').replace('B','X')
            test_seqs.append(seq)
            
        # Lưu ID để dùng cho file submission
        with open(TEST_IDS_PATH, 'wb') as f:
            pickle.dump(test_ids, f)
            
        extract_embeddings(test_seqs, MODEL_NAME, TEST_EMB_PATH)
        
    print("\n🎉 HOÀN TẤT TRÍCH XUẤT!")