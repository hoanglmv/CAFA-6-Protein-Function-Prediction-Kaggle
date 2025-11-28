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
    # Nếu file này nằm ở src/ver1, lùi lên 2 cấp để tới project root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Input test FASTA
    TEST_FASTA_PATH = os.path.join(BASE_DIR, 'data', 'Test', 'testsuperset.fasta')

    # Model & labels trong thư mục models/ver1 (theo yêu cầu)
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ver1')
    MODEL_PATH = None  # sẽ được xác định động phía dưới
    LABELS_MAP_PATH = os.path.join(MODEL_DIR, 'labels_map.pkl')

    # Output submission (lưu vào models/ver1)
    SUBMISSION_PATH = os.path.join(MODEL_DIR, 'submission.tsv')

    # Tham số (phải khớp khi train)
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

# Prepare GPU and ensure MODEL_DIR exists
TestConfig.setup_gpu()
os.makedirs(TestConfig.MODEL_DIR, exist_ok=True)

# Prefer best_model.keras then transformer_model.keras (legacy)
candidate_best = os.path.join(TestConfig.MODEL_DIR, 'best_model.keras')
candidate_final = os.path.join(TestConfig.MODEL_DIR, 'transformer_model.keras')
if os.path.exists(candidate_best):
    TestConfig.MODEL_PATH = candidate_best
elif os.path.exists(candidate_final):
    TestConfig.MODEL_PATH = candidate_final
else:
    # allow also 'transformer_model.h5' or top-level fallback for convenience
    alt_h5 = os.path.join(TestConfig.MODEL_DIR, 'transformer_model.h5')
    if os.path.exists(alt_h5):
        TestConfig.MODEL_PATH = alt_h5
    else:
        TestConfig.MODEL_PATH = None

# ==========================================
# 2. CUSTOM LAYERS (phải định nghĩa giống lúc train)
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
        # Load labels map
        print(f"[Inference] Loading Labels Map from {self.config.LABELS_MAP_PATH}...")
        if not os.path.exists(self.config.LABELS_MAP_PATH):
            raise FileNotFoundError(f"Labels map not found: {self.config.LABELS_MAP_PATH}")
        with open(self.config.LABELS_MAP_PATH, 'rb') as f:
            self.top_terms = pickle.load(f)
        print(f"   -> Loaded {len(self.top_terms)} GO terms.")

        # Load model
        if not self.config.MODEL_PATH:
            raise FileNotFoundError(f"No model found in {self.config.MODEL_DIR}. Put 'best_model.keras' or 'transformer_model.keras' there.")
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
            raise RuntimeError(f"Error loading model: {e}")

    def run_prediction(self):
        print(f"[Inference] Reading Test FASTA from {self.config.TEST_FASTA_PATH}...")
        if not os.path.exists(self.config.TEST_FASTA_PATH):
            raise FileNotFoundError(f"Test FASTA not found: {self.config.TEST_FASTA_PATH}")

        test_ids = []
        test_seqs = []
        for record in SeqIO.parse(self.config.TEST_FASTA_PATH, "fasta"):
            test_ids.append(record.id)
            test_seqs.append(str(record.seq))
        total_samples = len(test_ids)
        print(f"   -> Found {total_samples} test sequences.")

        # Ensure output dir exists
        out_dir = os.path.dirname(self.config.SUBMISSION_PATH)
        os.makedirs(out_dir, exist_ok=True)

        print(f"[Inference] Generating submission to {self.config.SUBMISSION_PATH}...")
        CHUNK_SIZE = 5000

        # Write results
        with open(self.config.SUBMISSION_PATH, 'w') as f:
            # Header (adjust if challenge requires different format)
            f.write("ObjectId\tGO-Term\tPrediction\n")

            for i in range(0, total_samples, CHUNK_SIZE):
                end_idx = min(i + CHUNK_SIZE, total_samples)
                print(f"   Processing chunk {i}..{end_idx-1} / {total_samples}", end='\r', flush=True)

                chunk_ids = test_ids[i:end_idx]
                chunk_seqs = test_seqs[i:end_idx]

                # Preprocess
                X_list = [[self.AA_MAP.get(aa, 0) for aa in seq] for seq in chunk_seqs]
                X_chunk = pad_sequences(X_list, maxlen=self.config.MAX_SEQ_LEN, padding='post', truncating='post')

                # Predict
                preds = self.model.predict(X_chunk, batch_size=self.config.BATCH_SIZE, verbose=0)

                # Write per sequence: only top-K (to keep file small)
                TOP_K = 60
                MIN_SCORE = 0.001

                for j, pid in enumerate(chunk_ids):
                    probs = preds[j]
                    top_indices = np.argsort(probs)[-TOP_K:]
                    # Sort descending for nicer output
                    top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]

                    for idx in top_indices:
                        score = float(probs[idx])
                        if score > MIN_SCORE:
                            term = self.top_terms[idx] if idx < len(self.top_terms) else f"TERM_{idx}"
                            f.write(f"{pid}\t{term}\t{score:.3f}\n")

        print(f"\n✅ DONE! Submission saved at:\n   {self.config.SUBMISSION_PATH}")

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    cfg = TestConfig()

    # Quick sanity checks and info
    print("Configuration:")
    print(f" - BASE_DIR: {cfg.BASE_DIR}")
    print(f" - TEST FASTA: {cfg.TEST_FASTA_PATH}")
    print(f" - MODEL DIR: {cfg.MODEL_DIR}")
    print(f" - MODEL PATH (chosen): {cfg.MODEL_PATH}")
    print(f" - LABELS MAP: {cfg.LABELS_MAP_PATH}")
    print(f" - SUBMISSION PATH: {cfg.SUBMISSION_PATH}")

    # Make sure required files exist
    if not os.path.exists(cfg.LABELS_MAP_PATH):
        print(f"❌ labels_map.pkl not found at {cfg.LABELS_MAP_PATH}. Run training first and copy labels_map.pkl into models/ver1.")
        exit(1)
    if not cfg.MODEL_PATH or not os.path.exists(cfg.MODEL_PATH):
        print(f"❌ Model not found in {cfg.MODEL_DIR}. Place 'best_model.keras' or 'transformer_model.keras' in models/ver1.")
        exit(1)
    if not os.path.exists(cfg.TEST_FASTA_PATH):
        print(f"❌ Test FASTA not found at {cfg.TEST_FASTA_PATH}. Place testsuperset.fasta into data/Test.")
        exit(1)

    engine = CAFA6Inference(cfg)
    engine.load_resources()
    engine.run_prediction()
