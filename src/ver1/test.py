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
    # Đường dẫn (Tự động lấy theo thư mục dự án)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    
    # Input
    TEST_FASTA_PATH = os.path.join(BASE_DIR, 'data', 'Test', 'testsuperset.fasta')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'transformer_model.keras')
    LABELS_MAP_PATH = os.path.join(BASE_DIR, 'models', 'labels_map.pkl')
    
    # Output
    SUBMISSION_PATH = os.path.join(BASE_DIR, 'submission.tsv')
    
    # Tham số (Phải khớp với lúc Train)
    MAX_SEQ_LEN = 512
    BATCH_SIZE = 64  # Batch size cho dự đoán
    
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
                print(e)
        else:
            print("⚠️ Running Inference on CPU")

TestConfig.setup_gpu()

# ==========================================
# 2. REDEFINE CUSTOM LAYERS (BẮT BUỘC)
# ==========================================
# Cần định nghĩa lại chính xác để Keras load được model
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
    def __init__(self, config):
        self.config = config
        self.model = None
        self.top_terms = []
        self.AA_MAP = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
        }

    def load_resources(self):
        print(f"[Inference] Loading Labels Map from {self.config.LABELS_MAP_PATH}...")
        try:
            with open(self.config.LABELS_MAP_PATH, 'rb') as f:
                self.top_terms = pickle.load(f)
            print(f"   -> Loaded {len(self.top_terms)} GO terms.")
        except FileNotFoundError:
            raise FileNotFoundError("❌ Không tìm thấy file labels_map.pkl! Bạn đã chạy train.py chưa?")

        print(f"[Inference] Loading Model from {self.config.MODEL_PATH}...")
        try:
            # Load model với Custom Objects
            self.model = tf.keras.models.load_model(
                self.config.MODEL_PATH,
                custom_objects={
                    "TransformerBlock": TransformerBlock,
                    "TokenAndPositionEmbedding": TokenAndPositionEmbedding
                }
            )
            print("   -> Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"❌ Lỗi load model: {e}")

    def run_prediction(self):
        print(f"[Inference] Reading Test FASTA from {self.config.TEST_FASTA_PATH}...")
        
        # 1. Đọc dữ liệu
        test_ids = []
        test_seqs = []
        # Lưu ý: CAFA test fasta thường có header đơn giản, lấy record.id là đủ
        for record in SeqIO.parse(self.config.TEST_FASTA_PATH, "fasta"):
            test_ids.append(record.id)
            test_seqs.append(str(record.seq))
            
        total_samples = len(test_ids)
        print(f"   -> Found {total_samples} test sequences.")
        
        # 2. Mở file để ghi (Stream writing)
        print(f"[Inference] Generating submission to {self.config.SUBMISSION_PATH}...")
        
        with open(self.config.SUBMISSION_PATH, 'w') as f:
            # Ghi Header (Kaggle CAFA thường không cần header hoặc header cụ thể, ở đây để mặc định)
            # Nếu Kaggle báo lỗi header, hãy comment dòng dưới lại.
            f.write("ObjectId\tGO-Term\tPrediction\n")
            
            # 3. Xử lý theo Batch (Chunk) để tránh tràn RAM
            CHUNK_SIZE = 5000
            
            for i in range(0, total_samples, CHUNK_SIZE):
                end_idx = min(i + CHUNK_SIZE, total_samples)
                print(f"   Processing chunk {i}/{total_samples}...", end='\r')
                
                # A. Lấy chunk hiện tại
                chunk_ids = test_ids[i:end_idx]
                chunk_seqs = test_seqs[i:end_idx]
                
                # B. Preprocess (Giống hệt lúc train)
                X_chunk_list = [[self.AA_MAP.get(aa, 0) for aa in seq] for seq in chunk_seqs]
                X_chunk = pad_sequences(X_chunk_list, maxlen=self.config.MAX_SEQ_LEN, padding='post', truncating='post')
                
                # C. Predict
                # verbose=0 để không in progress bar của Keras làm rối terminal
                preds = self.model.predict(X_chunk, batch_size=self.config.BATCH_SIZE, verbose=0)
                
                # D. Ghi kết quả
                for j, pid in enumerate(chunk_ids):
                    probs = preds[j]
                    
                    # --- CHIẾN THUẬT GHI FILE ---
                    # Không ghi tất cả 1500 term (file sẽ nặng vài GB).
                    # Chỉ ghi Top 50-100 term có điểm cao nhất.
                    
                    # Lấy index của 60 phần tử có xác suất cao nhất
                    top_indices = np.argsort(probs)[-60:] 
                    
                    for idx in top_indices:
                        score = probs[idx]
                        
                        # Chỉ ghi nếu score > 0.001 (0.1%) để loại bỏ nhiễu
                        if score > 0.001:
                            term = self.top_terms[idx]
                            # Định dạng TSV: ID <tab> Term <tab> Score
                            f.write(f"{pid}\t{term}\t{score:.3f}\n")
                            
        print(f"\n✅ DONE! File submission đã lưu tại:\n   {self.config.SUBMISSION_PATH}")

# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    cfg = TestConfig()
    
    # Kiểm tra file input tồn tại không
    if not os.path.exists(cfg.TEST_FASTA_PATH):
        print(f"❌ Error: Không tìm thấy file test tại {cfg.TEST_FASTA_PATH}")
        print("   Hãy đảm bảo bạn đã tải file 'testsuperset.fasta' vào thư mục data/Test")
        exit()

    engine = CAFA6Inference(cfg)
    engine.load_resources()
    engine.run_prediction()