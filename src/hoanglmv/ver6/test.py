import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH
# ==========================================
class Config:
    # Sửa lại số lượng dấu .. tùy vào vị trí bạn đặt file script
    # Giả sử file này nằm ở: src/hoanglmv/ver6/test.py -> lùi 3 cấp về Project Root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    # Input Data
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed2')
    TEST_FILE = os.path.join(DATA_DIR, 'test.parquet')
    
    # Model Directory (Nơi chứa model ver6 đã train)
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ver6')
    
    # Output Submission (Sửa: Lưu trực tiếp vào folder model)
    SUBMISSION_FILE = os.path.join(MODEL_DIR, 'submission.tsv')
    
    # Inference Params
    BATCH_SIZE = 256  # Batch lớn để chạy nhanh hơn (vì chỉ cần feed forward)
    TOP_K = 60        # Chỉ lấy 60 nhãn có điểm cao nhất
    MIN_SCORE = 0.001 # Chỉ lấy nhãn có xác suất > 0.1%

    @staticmethod
    def setup_gpu():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU Activated for Inference: {gpus[0].name}")
            except: pass
        else:
            print("⚠️ Running on CPU")

Config.setup_gpu()

# ==========================================
# 2. CUSTOM LOSS (Bắt buộc khai báo để Load Model)
# ==========================================
class AsymmetricLoss(tf.keras.losses.Loss):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def call(self, y_true, y_pred):
        return 0.0 # Không dùng khi test
    
    def get_config(self):
        return {'gamma_neg': self.gamma_neg, 'gamma_pos': self.gamma_pos, 'clip': self.clip, 'eps': self.eps}

# ==========================================
# 3. QUY TRÌNH DỰ ĐOÁN (INFERENCE)
# ==========================================
def run_inference():
    print(f"🚀 Bắt đầu Inference với model tại: {Config.MODEL_DIR}")
    
    # --- BƯỚC 1: LOAD TÀI NGUYÊN ---
    print("   -> Loading Model & Label Map...")
    
    # 1.1 Load Model
    model_path = os.path.join(Config.MODEL_DIR, 'best_model.keras')
    if not os.path.exists(model_path):
        # Fallback nếu không có best_model thì tìm final_model
        model_path = os.path.join(Config.MODEL_DIR, 'final_model.keras')
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Không tìm thấy model tại {Config.MODEL_DIR}")
        
    model = models.load_model(model_path, custom_objects={'AsymmetricLoss': AsymmetricLoss})
    
    # 1.2 Load Label Map (Index -> GO Term)
    map_path = os.path.join(Config.MODEL_DIR, 'idx_to_term.pkl')
    with open(map_path, 'rb') as f:
        idx_to_term = pickle.load(f)
    print(f"   -> Loaded mapping for {len(idx_to_term)} GO terms.")

    # --- BƯỚC 2: LOAD DỮ LIỆU TEST (PARQUET) ---
    print(f"   -> Reading Test File: {Config.TEST_FILE}")
    # Đọc file parquet bằng pandas (nhanh và tối ưu)
    df = pd.read_parquet(Config.TEST_FILE)
    
    ids = df['id'].values
    # Chuyển đổi list trong dataframe thành numpy matrix 2D
    # Lưu ý: Cần np.stack để biến mảng các object array thành mảng 2D chuẩn float32
    X_emb = np.stack(df['embedding'].values).astype(np.float32)
    X_tax = np.stack(df['superkingdom'].values).astype(np.float32)
    
    total_samples = len(ids)
    print(f"   -> Found {total_samples} samples.")
    
    # --- BƯỚC 3: DỰ ĐOÁN & GHI FILE ---
    print(f"   -> Predicting & Writing to {Config.SUBMISSION_FILE}...")
    
    # Đảm bảo thư mục tồn tại (thường model_dir đã có rồi, nhưng check cho chắc)
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    with open(Config.SUBMISSION_FILE, 'w') as f:
        # Ghi Header (theo chuẩn CAFA)
        f.write("ObjectId\tGO-Term\tPrediction\n")
        
        # Xử lý theo batch để tiết kiệm RAM
        # tqdm giúp hiện thanh tiến trình
        for i in tqdm(range(0, total_samples, Config.BATCH_SIZE), desc="Processing Batches"):
            end = min(i + Config.BATCH_SIZE, total_samples)
            
            # Lấy batch hiện tại
            batch_emb = X_emb[i:end]
            batch_tax = X_tax[i:end]
            batch_ids = ids[i:end]
            
            # Predict (Đầu vào là Dictionary khớp với tên Layer trong train.py)
            preds = model.predict(
                {'input_embedding': batch_emb, 'input_taxonomy': batch_tax}, 
                verbose=0
            )
            
            # Xử lý kết quả batch
            for j, pid in enumerate(batch_ids):
                probs = preds[j]
                
                # Chiến thuật lọc: Chỉ lấy Top K điểm cao nhất
                # np.argsort trả về index của phần tử được sort tăng dần -> lấy [-TOP_K:] -> đảo ngược [::-1]
                top_indices = np.argsort(probs)[-Config.TOP_K:][::-1]
                
                for idx in top_indices:
                    score = float(probs[idx])
                    
                    # Chỉ ghi nếu điểm > ngưỡng tối thiểu
                    if score > Config.MIN_SCORE:
                        term = idx_to_term.get(idx, None)
                        if term:
                            f.write(f"{pid}\t{term}\t{score:.3f}\n")
                        
    print(f"\n✅ HOÀN TẤT! File submission đã lưu tại: {Config.SUBMISSION_FILE}")

if __name__ == "__main__":
    run_inference()