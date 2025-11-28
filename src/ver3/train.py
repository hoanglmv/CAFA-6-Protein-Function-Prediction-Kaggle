import os
import pickle
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from IPython.display import display, clear_output

# ==========================================
# 1. CONFIGURATION (CẤU HÌNH)
# ==========================================
class Config:
    # Đường dẫn gốc
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Input Data (Lưu ý: Đảm bảo process_data.py đã chạy và lưu file này)
    # Ta dùng ver2 hoặc ver3 tùy bạn đặt tên folder
    DATA_PATH = os.path.join(BASE_DIR, 'models', 'ver3', 'processed_data.pkl')
    
    # Output Model
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ver3')
    
    # Model Parameters
    MAX_SEQ_LEN = 512
    NUM_CLASSES = 1500
    VOCAB_SIZE  = 21    # Amino Acids
    TAX_EMBED_DIM = 16  # Kích thước vector nhúng cho loài (Taxonomy)
    
    # Training Parameters
    BATCH_SIZE  = 64    # RTX 3060 12GB
    EPOCHS      = 30
    LEARNING_RATE = 1e-3
    EMBED_DIM   = 64
    NUM_HEADS   = 4
    FF_DIM      = 128
    
    # Callbacks
    EARLY_STOP_PATIENCE = 5
    REDUCE_LR_PATIENCE  = 3
    
    @staticmethod
    def setup_gpu():
        print("[Setup] Checking GPU...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU Activated: {gpus[0].name}")
            except RuntimeError as e:
                print(e)
        else:
            print("⚠️ Running on CPU")
            
    def save_to_json(self):
        # Lọc bỏ các thuộc tính không serializable
        config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)}
        path = os.path.join(self.MODEL_DIR, 'config.json')
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"[Config] Saved configuration to {path}")

# Setup môi trường
Config.setup_gpu()
os.makedirs(Config.MODEL_DIR, exist_ok=True)

# ==========================================
# 2. DATASET (XỬ LÝ SEQUENCE + TAXONOMY)
# ==========================================
class CAFA6Dataset:
    def __init__(self, data_path, max_len=512, num_classes=1500):
        self.data_path = data_path
        self.max_len = max_len
        self.num_classes = num_classes
        
        # Mapping Amino Acid
        self.AA_MAP = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
        }
        
        # Variables for Mappings
        self.top_terms = []
        self.term_to_idx = {}
        self.tax_to_idx = {} # Map loài 9606 -> index 1
        self.num_taxons = 0  # Tổng số loài tìm thấy
        
        self.df = self.load_data()
        # X_seq: Sequence, X_tax: Taxonomy, Y: Labels
        self.X_seq, self.X_tax, self.Y = self.process_data()

    def load_data(self):
        print(f"[Dataset] Loading .pkl file from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ File not found: {self.data_path}")
        return pd.read_pickle(self.data_path)

    def process_data(self):
        print(f"[Dataset] Processing {len(self.df)} samples...")
        
        # --- 1. PROCESS SEQUENCE ---
        print("   -> Tokenizing Sequences...")
        X_seq_list = [[self.AA_MAP.get(aa, 0) for aa in seq] for seq in self.df['sequence']]
        X_seq = pad_sequences(X_seq_list, maxlen=self.max_len, padding='post', truncating='post')
        
        # --- 2. PROCESS TAXONOMY (MỚI) ---
        print("   -> Encoding Taxonomy...")
        # Lấy danh sách các loài duy nhất
        unique_taxons = self.df['taxonomyID'].unique()
        # Tạo từ điển map: {9606: 1, 10090: 2, ...} (0 dành cho unknown/padding)
        self.tax_to_idx = {tax: i+1 for i, tax in enumerate(unique_taxons)}
        self.num_taxons = len(unique_taxons) + 1 # +1 cho số 0
        
        # Chuyển cột taxonomyID thành index
        X_tax = np.array([self.tax_to_idx.get(t, 0) for t in self.df['taxonomyID']])
        
        # --- 3. PROCESS LABELS ---
        print("   -> Encoding Labels...")
        all_terms = [t for sublist in self.df['term'] for t in sublist]
        self.top_terms = [t[0] for t in Counter(all_terms).most_common(self.num_classes)]
        self.term_to_idx = {t: i for i, t in enumerate(self.top_terms)}
        
        Y = np.zeros((len(self.df), self.num_classes), dtype='float32')
        for i, terms in enumerate(self.df['term']):
            for t in terms:
                if t in self.term_to_idx:
                    Y[i, self.term_to_idx[t]] = 1.0
                    
        print(f"[Dataset] Shapes -> Seq: {X_seq.shape}, Tax: {X_tax.shape}, Y: {Y.shape}")
        print(f"[Dataset] Found {self.num_taxons} unique species.")
        return X_seq, X_tax, Y
    
    def save_mappings(self, model_dir):
        # Lưu GO Terms Map
        with open(os.path.join(model_dir, 'labels_map.pkl'), 'wb') as f:
            pickle.dump(self.top_terms, f)
        # Lưu Taxonomy Map (Rất quan trọng cho file Test)
        with open(os.path.join(model_dir, 'tax_map.pkl'), 'wb') as f:
            pickle.dump(self.tax_to_idx, f)
        print(f"[Dataset] Saved mappings to {model_dir}")

# ==========================================
# 3. DATAMODULE (MULTI-INPUT HANDLING)
# ==========================================
class CAFA6DataModule:
    def __init__(self, dataset, batch_size=64, test_size=0.1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_size = test_size
        self.train_ds = None
        self.val_ds = None
        # Lưu dữ liệu val thô để evaluate thủ công
        self.val_data = {} 

    def setup(self):
        print("[DataModule] Splitting Train/Val...")
        # Split 3 arrays: Sequence, Taxonomy, Labels
        X_seq_train, X_seq_val, X_tax_train, X_tax_val, y_train, y_val = train_test_split(
            self.dataset.X_seq, self.dataset.X_tax, self.dataset.Y, 
            test_size=self.test_size, random_state=42
        )
        
        # Store validation data specifically for manual predict
        self.val_data = {
            'seq': X_seq_val,
            'tax': X_tax_val,
            'y': y_val
        }
        
        # Create Dictionary Inputs for Keras (Tạo dict để map với layer name)
        train_inputs = {'seq_input': X_seq_train, 'tax_input': X_tax_train}
        val_inputs   = {'seq_input': X_seq_val,   'tax_input': X_tax_val}
        
        print(f"   -> Train samples: {len(X_seq_train)}")
        print(f"   -> Val samples:   {len(X_seq_val)}")
        
        # TF Dataset pipeline
        self.train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, y_train))\
            .shuffle(1024).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            
        self.val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, y_val))\
            .batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 4. MODEL ARCHITECTURE (DUAL INPUT)
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

class ProteinMultiInputModel:
    def __init__(self, config, num_taxons):
        self.config = config
        self.num_taxons = num_taxons # Số lượng loài phát hiện được từ Dataset
        self.model = self._build_model()
        
    def _build_model(self):
        # --- BRANCH 1: SEQUENCE ---
        seq_input = layers.Input(shape=(self.config.MAX_SEQ_LEN,), name='seq_input')
        x1 = TokenAndPositionEmbedding(self.config.MAX_SEQ_LEN, self.config.VOCAB_SIZE, self.config.EMBED_DIM)(seq_input)
        x1 = TransformerBlock(self.config.EMBED_DIM, self.config.NUM_HEADS, self.config.FF_DIM)(x1)
        x1 = layers.GlobalAveragePooling1D()(x1)
        
        # --- BRANCH 2: TAXONOMY ---
        tax_input = layers.Input(shape=(1,), name='tax_input')
        # Embedding loài: (Batch, 1, 16)
        x2 = layers.Embedding(input_dim=self.num_taxons, output_dim=self.config.TAX_EMBED_DIM)(tax_input)
        x2 = layers.Flatten()(x2) # (Batch, 16)
        
        # --- MERGE BRANCHES ---
        x = layers.Concatenate()([x1, x2]) # Nối vector đặc trưng
        
        # --- HEAD ---
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(self.config.NUM_CLASSES, activation="sigmoid")(x)
        
        model = models.Model(inputs=[seq_input, tax_input], outputs=outputs, name="ProteinMultiInput")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["binary_accuracy", tf.keras.metrics.AUC(multi_label=True, name='auc')]
        )
        return model

    def train(self, data_module, callbacks_list):
        print(f"\n[Model] Start Training ({self.config.EPOCHS} Epochs)...")
        return self.model.fit(
            data_module.train_ds,
            validation_data=data_module.val_ds,
            epochs=self.config.EPOCHS,
            callbacks=callbacks_list,
            verbose=1
        )

# ==========================================
# 5. UTILS
# ==========================================
def plot_history(history, save_dir):
    pd.DataFrame(history.history).to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title('Loss')
    ax[0].legend(); ax[0].grid(True)
    
    ax[1].plot(history.history['auc'], label='Train AUC')
    ax[1].plot(history.history['val_auc'], label='Val AUC')
    ax[1].set_title('AUC')
    ax[1].legend(); ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_plot.png'))
    plt.show()

def predict_sample(model_wrapper, dm, dataset, num_samples=3):
    print("\n=== SAMPLE PREDICTION ===")
    # Lấy data thô từ val_data (Dictionary)
    X_sample_seq = dm.val_data['seq'][:num_samples]
    X_sample_tax = dm.val_data['tax'][:num_samples]
    y_true = dm.val_data['y'][:num_samples]
    
    # Predict cần dictionary đầu vào
    preds = model_wrapper.model.predict({'seq_input': X_sample_seq, 'tax_input': X_sample_tax})
    
    for i in range(num_samples):
        print(f"\nSample #{i+1}:")
        true_idxs = np.where(y_true[i] == 1)[0]
        true_terms = [dataset.top_terms[k] for k in true_idxs]
        
        pred_idxs = np.where(preds[i] > 0.2)[0]
        pred_terms = [dataset.top_terms[k] for k in pred_idxs]
        
        print(f"  Target: {true_terms[:5]} ...")
        print(f"  Predict: {pred_terms[:5]} ...")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    cfg = Config()
    
    # 1. Load Dataset
    dataset = CAFA6Dataset(cfg.DATA_PATH, cfg.MAX_SEQ_LEN, cfg.NUM_CLASSES)
    # Lưu cả 2 maps: Label Map + Tax Map
    dataset.save_mappings(cfg.MODEL_DIR)
    
    # 2. DataModule
    dm = CAFA6DataModule(dataset, cfg.BATCH_SIZE)
    dm.setup()
    
    # 3. Callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(os.path.join(cfg.MODEL_DIR, 'best_model.keras'), monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=cfg.EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=cfg.REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1),
        callbacks.CSVLogger(os.path.join(cfg.MODEL_DIR, 'training_log.csv'))
    ]
    
    # 4. Train with Multi-Input Model
    # Truyền số lượng loài vào để khởi tạo lớp Embedding
    num_taxons = dataset.num_taxons
    print(f"\n[Model] Initializing for {num_taxons} species...")
    
    with tf.device('/GPU:0'):
        model_wrapper = ProteinMultiInputModel(cfg, num_taxons)
        model_wrapper.model.summary()
        
        history = model_wrapper.train(dm, callbacks_list)
        
        model_wrapper.model.save(os.path.join(cfg.MODEL_DIR, 'final_model.keras'))
        
    # 5. Evaluate
    cfg.save_to_json()
    plot_history(history, cfg.MODEL_DIR)
    
    print("\n=== FINAL EVALUATION ===")
    val_metrics = model_wrapper.model.evaluate(dm.val_ds)
    print(f"Loss: {val_metrics[0]:.4f} | Accuracy: {val_metrics[1]:.4f} | AUC: {val_metrics[2]:.4f}")
    
    predict_sample(model_wrapper, dm, dataset)