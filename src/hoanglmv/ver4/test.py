import os
import re
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
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

    # Input Test Data
    TEST_FASTA_PATH = os.path.join(BASE_DIR, 'data', 'Test', 'testsuperset.fasta')

    # --- CẤU HÌNH VER 4 ---
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ver4')
    
    # Files Mappings (Cần cả Label và Tax)
    LABELS_MAP_PATH = os.path.join(MODEL_DIR, 'labels_map.pkl')
    TAX_MAP_PATH    = os.path.join(MODEL_DIR, 'tax_map.pkl')

    # Output Submission
    SUBMISSION_PATH = os.path.join(BASE_DIR, 'submission.tsv')

    # Model Path (Sẽ tự động tìm)
    MODEL_PATH = None 

    # Params (Phải khớp với Train)
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
            print("⚠️ Running Inference on CPU")

TestConfig.setup_gpu()

# Tự động chọn model tốt nhất trong ver4
candidates = [
    os.path.join(TestConfig.MODEL_DIR, 'best_model.keras'),
    os.path.join(TestConfig.MODEL_DIR, 'final_model.keras')
]
for path in candidates:
    if os.path.exists(path):
        TestConfig.MODEL_PATH = path
        print(f"🎯 Selected Model: {os.path.basename(path)}")
        break

if not TestConfig.MODEL_PATH:
    print("❌ Critical Error: Không tìm thấy model nào trong thư mục models/ver4!")
    exit(1)

# ==========================================
# 2. CUSTOM LAYERS & LOSS (BẮT BUỘC KHAI BÁO)
# ==========================================
# Phải copy y nguyên Class AsymmetricLoss từ train.py sang đây
class AsymmetricLoss(tf.keras.losses.Loss):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, **kwargs):
        super(AsymmetricLoss, self).__init__(**kwargs)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def call(self, y_true, y_pred):
        # Hàm call không dùng khi inference, nhưng cần để load model không bị lỗi
        return 0.0 
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma_neg": self.gamma_neg,
            "gamma_pos": self.gamma_pos,
            "clip": self.clip,
            "eps": self.eps
        })
        return config

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
# 3. INFERENCE ENGINE (MULTI-INPUT)
# ==========================================
class CAFA6Inference:
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.top_terms = []
        self.tax_map = {} # Map lưu TaxID -> Index
        
        self.AA_MAP = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
        }

    def load_resources(self):
        print(f"[Inference] Loading Resources from {self.config.MODEL_DIR}...")
        
        # 1. Labels Map
        if not os.path.exists(self.config.LABELS_MAP_PATH):
            raise FileNotFoundError(f"Missing labels_map.pkl at {self.config.LABELS_MAP_PATH}")
        with open(self.config.LABELS_MAP_PATH, 'rb') as f:
            self.top_terms = pickle.load(f)
        
        # 2. Taxonomy Map (Quan trọng cho Ver4)
        if os.path.exists(self.config.TAX_MAP_PATH):
            with open(self.config.TAX_MAP_PATH, 'rb') as f:
                self.tax_map = pickle.load(f)
            print(f"   -> Loaded Tax Map with {len(self.tax_map)} species.")
        else:
            print("⚠️ Warning: Tax Map not found! All taxonomy inputs will be 0.")
            self.tax_map = {}

        # 3. Load Model with Custom Objects
        print(f"[Inference] Loading Model: {self.config.MODEL_PATH}...")
        self.model = tf.keras.models.load_model(
            self.config.MODEL_PATH,
            custom_objects={
                "TransformerBlock": TransformerBlock,
                "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                "AsymmetricLoss": AsymmetricLoss # Thêm Class Loss vào đây
            }
        )
        print("   -> Model loaded successfully.")

    def run_prediction(self):
        print(f"[Inference] Parsing Test FASTA: {self.config.TEST_FASTA_PATH}")
        
        # Đọc dữ liệu
        test_ids = []
        test_seqs = []
        test_taxs = []
        
        for record in SeqIO.parse(self.config.TEST_FASTA_PATH, "fasta"):
            test_ids.append(record.id)
            test_seqs.append(str(record.seq))
            
            # Trích xuất Taxonomy ID từ Header (VD: OX=9606)
            ox_match = re.search(r'OX=(\d+)', record.description)
            if ox_match:
                raw_tax_id = int(ox_match.group(1))
                # Map sang index mà model hiểu (nếu không có thì = 0)
                mapped_tax = self.tax_map.get(raw_tax_id, 0)
            else:
                mapped_tax = 0
            test_taxs.append(mapped_tax)
            
        total_samples = len(test_ids)
        print(f"   -> Found {total_samples} test sequences.")

        # Tạo output folder
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)
        print(f"[Inference] Writing to {self.config.SUBMISSION_PATH}...")
        
        CHUNK_SIZE = 5000

        with open(self.config.SUBMISSION_PATH, 'w') as f:
            # Header chuẩn của CAFA Submission
            f.write("ObjectId\tGO-Term\tPrediction\n")

            for i in range(0, total_samples, CHUNK_SIZE):
                end_idx = min(i + CHUNK_SIZE, total_samples)
                print(f"   Processing {i}..{end_idx-1}...", end='\r', flush=True)

                # --- PREPARE BATCH ---
                chunk_ids = test_ids[i:end_idx]
                chunk_seqs = test_seqs[i:end_idx]
                chunk_taxs = test_taxs[i:end_idx]

                # 1. Process Sequence
                X_list = [[self.AA_MAP.get(aa, 0) for aa in seq] for seq in chunk_seqs]
                X_seq_chunk = pad_sequences(X_list, maxlen=self.config.MAX_SEQ_LEN, padding='post', truncating='post')

                # 2. Process Taxonomy (Numpy array)
                X_tax_chunk = np.array(chunk_taxs)

                # --- PREDICT (MULTI-INPUT) ---
                # Key 'seq_input' và 'tax_input' phải khớp với layer name trong train.py
                preds = self.model.predict(
                    {'seq_input': X_seq_chunk, 'tax_input': X_tax_chunk}, 
                    batch_size=self.config.BATCH_SIZE, 
                    verbose=0
                )

                # --- WRITE RESULTS ---
                TOP_K = 60
                MIN_SCORE = 0.001 # Ngưỡng lọc bớt rác

                for j, pid in enumerate(chunk_ids):
                    probs = preds[j]
                    # Lấy Top K index có điểm cao nhất
                    top_indices = np.argsort(probs)[-TOP_K:]
                    # Sắp xếp giảm dần
                    top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]

                    for idx in top_indices:
                        score = float(probs[idx])
                        if score > MIN_SCORE:
                            term = self.top_terms[idx]
                            f.write(f"{pid}\t{term}\t{score:.3f}\n")

        print(f"\n✅ DONE! Submission ready at: {self.config.SUBMISSION_PATH}")

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    cfg = TestConfig()
    
    if not os.path.exists(cfg.TEST_FASTA_PATH):
        print(f"❌ Error: Test FASTA not found at {cfg.TEST_FASTA_PATH}")
        exit(1)

    engine = CAFA6Inference(cfg)
    engine.load_resources()
    engine.run_prediction()