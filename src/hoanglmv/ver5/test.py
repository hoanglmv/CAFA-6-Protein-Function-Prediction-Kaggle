import os
import re
import pickle
import numpy as np
import pandas as pd # Cần thêm pandas để đọc file map
import tensorflow as tf
from Bio import SeqIO
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# 1. CONFIGURATION
# ==========================================
class TestConfig:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    TEST_FASTA_PATH = os.path.join(BASE_DIR, 'data', 'Test', 'testsuperset.fasta')
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ver4')
    
    # Các file cần thiết
    LABELS_MAP_PATH = os.path.join(MODEL_DIR, 'labels_map.pkl')
    TAX_MAP_PATH    = os.path.join(MODEL_DIR, 'tax_map.pkl') # Map ID -> Index (0,1,2)
    
    # FILE MỚI: Map ID Lạ -> ID Quen
    TAX_CONVERSION_PATH = os.path.join(MODEL_DIR, 'taxonomy_mapping.tsv') 

    SUBMISSION_PATH = os.path.join(BASE_DIR, 'submission.tsv')
    MODEL_PATH = None 
    MAX_SEQ_LEN = 512
    BATCH_SIZE = 64

    @staticmethod
    def setup_gpu():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU: {gpus[0].name}")
            except: pass

TestConfig.setup_gpu()

# Auto-select model
candidates = [os.path.join(TestConfig.MODEL_DIR, 'best_model.keras'), os.path.join(TestConfig.MODEL_DIR, 'final_model.keras')]
for path in candidates:
    if os.path.exists(path):
        TestConfig.MODEL_PATH = path
        print(f"🎯 Model: {os.path.basename(path)}")
        break

# ==========================================
# 2. CUSTOM OBJECTS (Keep same as Train)
# ==========================================
class AsymmetricLoss(tf.keras.losses.Loss):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred): return 0.0
    def get_config(self): return {}

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.num_heads, self.ff_dim, self.rate = embed_dim, num_heads, ff_dim, rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6); self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate); self.dropout2 = layers.Dropout(rate)
    def call(self, inputs, training=False):
        attn_out = self.att(inputs, inputs); out1 = self.layernorm1(inputs + self.dropout1(attn_out, training=training))
        return self.layernorm2(out1 + self.dropout2(self.ffn(out1), training=training))
    def get_config(self): return {"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate}

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen, self.vocab_size, self.embed_dim = maxlen, vocab_size, embed_dim
        self.token_emb = layers.Embedding(vocab_size, embed_dim); self.pos_emb = layers.Embedding(maxlen, embed_dim)
    def call(self, x): return self.token_emb(x) + self.pos_emb(tf.range(start=0, limit=tf.shape(x)[-1], delta=1))
    def get_config(self): return {"maxlen": self.maxlen, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim}

# ==========================================
# 3. INFERENCE ENGINE (UPDATED)
# ==========================================
class CAFA6Inference:
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.top_terms = []
        self.tax_index_map = {} # Map: TaxID -> 0, 1, 2 (Model Index)
        self.tax_conversion_dict = {} # Map: UnknownID -> KnownID
        
        self.AA_MAP = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                       'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

    def load_resources(self):
        print(f"[Inference] Loading Resources...")
        # 1. Labels
        with open(self.config.LABELS_MAP_PATH, 'rb') as f: self.top_terms = pickle.load(f)
        
        # 2. Tax Index Map (Train ID -> Model Index)
        with open(self.config.TAX_MAP_PATH, 'rb') as f: self.tax_index_map = pickle.load(f)
        
        # 3. Tax Conversion Map (Unknown ID -> Train ID) - MỚI
        if os.path.exists(self.config.TAX_CONVERSION_PATH):
            df_map = pd.read_csv(self.config.TAX_CONVERSION_PATH, sep='\t', dtype=str)
            # Tạo dict: {'Original_ID': 'Mapped_ID'}
            self.tax_conversion_dict = dict(zip(df_map['Original_ID'], df_map['Mapped_ID']))
            print(f"   -> Loaded {len(self.tax_conversion_dict)} taxonomy conversion rules.")
        else:
            print("⚠️ Warning: Taxonomy Mapping file not found. Accuracy will be low on unseen species.")

        # 4. Model
        self.model = tf.keras.models.load_model(self.config.MODEL_PATH, custom_objects={
            "TransformerBlock": TransformerBlock, "TokenAndPositionEmbedding": TokenAndPositionEmbedding, "AsymmetricLoss": AsymmetricLoss})

    def run_prediction(self):
        print(f"[Inference] Parsing Test FASTA...")
        test_ids, test_seqs, test_taxs = [], [], []
        
        for record in SeqIO.parse(self.config.TEST_FASTA_PATH, "fasta"):
            test_ids.append(record.id)
            test_seqs.append(str(record.seq))
            
            # Parsing Header: ">ID TaxID"
            parts = record.description.split()
            raw_tax_id = parts[1] if len(parts) >= 2 and parts[1].isdigit() else "0"
            
            # --- LOGIC QUAN TRỌNG NHẤT ---
            # Bước 1: Thử map loài lạ về loài quen
            mapped_tax_id = self.tax_conversion_dict.get(raw_tax_id, raw_tax_id)
            
            # Bước 2: Map loài quen về Index của model (0, 1, 2...)
            # Lưu ý: Cần chuyển sang int vì tax_index_map lưu key là int (từ training)
            try:
                tax_idx = self.tax_index_map.get(int(mapped_tax_id), 0)
            except:
                tax_idx = 0 # Fallback về 0 nếu có lỗi
                
            test_taxs.append(tax_idx)
            
        total = len(test_ids)
        print(f"   -> Found {total} sequences.")
        
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)
        print(f"[Inference] Generating submission...")
        
        CHUNK_SIZE = 2000 # Batch size an toàn
        with open(self.config.SUBMISSION_PATH, 'w') as f:
            f.write("ObjectId\tGO-Term\tPrediction\n")
            
            for i in range(0, total, CHUNK_SIZE):
                end = min(i + CHUNK_SIZE, total)
                print(f"   Processing {i}..{end}...", end='\r')
                
                # Batch Data
                chunk_ids = test_ids[i:end]
                X_seq = pad_sequences([[self.AA_MAP.get(aa, 0) for aa in seq] for seq in test_seqs[i:end]], 
                                      maxlen=self.config.MAX_SEQ_LEN, padding='post', truncating='post')
                X_tax = np.array(test_taxs[i:end])
                
                # Predict
                preds = self.model.predict({'seq_input': X_seq, 'tax_input': X_tax}, batch_size=self.config.BATCH_SIZE, verbose=0)
                
                # Write High Scores
                for j, pid in enumerate(chunk_ids):
                    probs = preds[j]
                    top_k = np.argsort(probs)[-60:][::-1] # Top 60
                    for idx in top_k:
                        score = float(probs[idx])
                        if score > 0.001:
                            f.write(f"{pid}\t{self.top_terms[idx]}\t{score:.3f}\n")
                            
        print(f"\n✅ DONE! File saved at: {self.config.SUBMISSION_PATH}")

if __name__ == "__main__":
    cfg = TestConfig()
    engine = CAFA6Inference(cfg)
    engine.load_resources()
    engine.run_prediction()