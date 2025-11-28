import os
import pickle
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# 1. CONFIGURATION
# ==========================================
class TestConfig:
    # Lùi 2 cấp từ src/ver2 về Project Root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Input Test Data
    TEST_FASTA_PATH = os.path.join(BASE_DIR, 'data', 'Test', 'testsuperset.fasta')

    # --- CẬP NHẬT QUAN TRỌNG: Dùng ver2 (nơi bạn vừa train xong) ---
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ver2')
    
    # Đường dẫn file mapping nhãn
    LABELS_MAP_PATH = os.path.join(MODEL_DIR, 'labels_map.pkl')

    # Output Submission (Lưu ngay tại Project Root hoặc trong models/ver2)
    SUBMISSION_PATH = os.path.join(BASE_DIR, 'submission.tsv')

    # Model Path (Sẽ tự động tìm file tốt nhất)
    MODEL_PATH = None 

    # Tham số (Phải khớp với lúc Train)
    MAX_SEQ_LEN = 512
    BATCH_SIZE = 64

    # Setup GPU
    @staticmethod
    def setup_gpu():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU Activated for Inference: {gpus[0].name}")
            except RuntimeError as e:
                print("GPU setup error:", e)
        else:
            print("⚠️ Running Inference on CPU (Sẽ rất chậm!)")

# Khởi chạy cấu hình
TestConfig.setup_gpu()

# Tự động chọn model tốt nhất
# Ưu tiên 1: best_model.keras (Model có Val AUC cao nhất)
# Ưu tiên 2: final_model.keras (Model ở epoch cuối cùng)
candidate_best = os.path.join(TestConfig.MODEL_DIR, 'best_model.keras')
candidate_final = os.path.join(TestConfig.MODEL_DIR, 'final_model.keras')
candidate_legacy = os.path.join(TestConfig.MODEL_DIR, 'transformer_model.keras')

if os.path.exists(candidate_best):
    TestConfig.MODEL_PATH = candidate_best
    print(f"🎯 Selected Model: best_model.keras")
elif os.path.exists(candidate_final):
    TestConfig.MODEL_PATH = candidate_final
    print(f"🎯 Selected Model: final_model.keras")
elif os.path.exists(candidate_legacy):
    TestConfig.MODEL_PATH = candidate_legacy
    print(f"🎯 Selected Model: transformer_model.keras")
else:
    TestConfig.MODEL_PATH = None

# ==========================================
# 2. CUSTOM LAYERS (BẮT BUỘC ĐỊNH NGHĨA LẠI)
# ==========================================
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return config

# ==========================================
# 3. INFERENCE ENGINE
# ==========================================
class CAFA6Inference:
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.top_terms = []
        self.AA_MAP = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
        }

    def load_resources(self):
        # 1. Load Labels Map
        print(f"[Inference] Loading Labels Map from {self.config.LABELS_MAP_PATH}...")
        if not os.path.exists(self.config.LABELS_MAP_PATH):
            raise FileNotFoundError(f"❌ Labels map not found: {self.config.LABELS_MAP_PATH}")
        
        with open(self.config.LABELS_MAP_PATH, 'rb') as f:
            self.top_terms = pickle.load(f)
        print(f"   -> Loaded {len(self.top_terms)} GO terms.")

        # 2. Load Model
        if not self.config.MODEL_PATH:
            raise FileNotFoundError(f"❌ No model found in {self.config.MODEL_DIR}")
        
        print(f"[Inference] Loading Model from {self.config.MODEL_PATH}...")
        try:
            self.model = tf.keras.models.load_model(
                self.config.MODEL_PATH,
                custom_objects={
                    "TransformerBlock": TransformerBlock,
                    "TokenAndPositionEmbedding": TokenAndPositionEmbedding
                }
            )
            print("   -> Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"❌ Error loading model: {e}")

    def run_prediction(self):
        print(f"[Inference] Reading Test FASTA from {self.config.TEST_FASTA_PATH}...")
        if not os.path.exists(self.config.TEST_FASTA_PATH):
            raise FileNotFoundError(f"❌ Test FASTA not found: {self.config.TEST_FASTA_PATH}")

        # Đọc dữ liệu Test
        test_ids = []
        test_seqs = []
        for record in SeqIO.parse(self.config.TEST_FASTA_PATH, "fasta"):
            test_ids.append(record.id)
            test_seqs.append(str(record.seq))
            
        total_samples = len(test_ids)
        print(f"   -> Found {total_samples} test sequences.")

        # Tạo thư mục output nếu chưa có
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)

        print(f"[Inference] Generating submission to {self.config.SUBMISSION_PATH}...")
        CHUNK_SIZE = 5000 # Xử lý từng cụm 5000 mẫu để tiết kiệm RAM

        with open(self.config.SUBMISSION_PATH, 'w') as f:
            # Ghi Header chuẩn Kaggle
            f.write("ObjectId\tGO-Term\tPrediction\n")

            for i in range(0, total_samples, CHUNK_SIZE):
                end_idx = min(i + CHUNK_SIZE, total_samples)
                print(f"   Processing chunk {i}..{end_idx-1} / {total_samples}...", end='\r', flush=True)

                # A. Lấy chunk
                chunk_ids = test_ids[i:end_idx]
                chunk_seqs = test_seqs[i:end_idx]

                # B. Preprocess (Tokenize + Pad)
                X_list = [[self.AA_MAP.get(aa, 0) for aa in seq] for seq in chunk_seqs]
                X_chunk = pad_sequences(X_list, maxlen=self.config.MAX_SEQ_LEN, padding='post', truncating='post')

                # C. Predict
                preds = self.model.predict(X_chunk, batch_size=self.config.BATCH_SIZE, verbose=0)

                # D. Write Results (Lọc Top K)
                TOP_K = 60
                MIN_SCORE = 0.001 # Chỉ lấy xác suất > 0.1%

                for j, pid in enumerate(chunk_ids):
                    probs = preds[j]
                    
                    # Lấy index của các xác suất cao nhất
                    top_indices = np.argsort(probs)[-TOP_K:]
                    # Sắp xếp giảm dần (cao nhất đứng đầu)
                    top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]

                    for idx in top_indices:
                        score = float(probs[idx])
                        if score > MIN_SCORE:
                            # Mapping ngược từ Index -> GO Term
                            term = self.top_terms[idx]
                            f.write(f"{pid}\t{term}\t{score:.3f}\n")

        print(f"\n✅ DONE! Submission saved at:\n   {self.config.SUBMISSION_PATH}")

# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    cfg = TestConfig()

    print("\n=== CONFIGURATION CHECK ===")
    print(f" - BASE_DIR:       {cfg.BASE_DIR}")
    print(f" - MODEL_DIR:      {cfg.MODEL_DIR}")
    print(f" - CHOSEN MODEL:   {cfg.MODEL_PATH}")
    print(f" - SUBMISSION:     {cfg.SUBMISSION_PATH}")
    
    if cfg.MODEL_PATH is None:
        print("❌ CRITICAL: Không tìm thấy model nào trong thư mục models/ver2!")
        print("   Hãy chắc chắn bạn đã chạy train.py thành công.")
        exit(1)

    engine = CAFA6Inference(cfg)
    engine.load_resources()
    engine.run_prediction()