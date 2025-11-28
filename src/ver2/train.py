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
# 1. CONFIGURATION (CẤU HÌNH - ĐÃ SỬA LỖI)
# ==========================================
class Config:
    # --- SỬA LỖI QUAN TRỌNG ---
    # Bỏ dấu nháy '' quanh __file__ để Python lấy đúng đường dẫn file script
    # src/ver2/train.py -> lùi 2 cấp -> Project Root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Cập nhật đường dẫn cho khớp với output của process_data.py (ver2)
    DATA_PATH = os.path.join(BASE_DIR, 'models', 'ver2', 'processed_data.pkl')
    
    # Nơi lưu Model checkpoints (cũng update sang ver2)
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ver2')
    
    # Model Parameters
    MAX_SEQ_LEN = 512
    NUM_CLASSES = 1500
    VOCAB_SIZE  = 21    # 20 Axit amin + 1 Padding
    
    # Training Parameters
    BATCH_SIZE  = 64    # RTX 3060 12GB
    EPOCHS      = 30
    LEARNING_RATE = 1e-3
    EMBED_DIM   = 64
    NUM_HEADS   = 4
    FF_DIM      = 128
    
    # Callbacks Settings
    EARLY_STOP_MONITOR = 'val_auc' # Theo dõi AUC thay vì Loss để tối ưu hơn
    EARLY_STOP_PATIENCE = 5
    REDUCE_LR_PATIENCE  = 3
    REDUCE_LR_FACTOR    = 0.5
    MIN_LR              = 1e-6
    
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
        config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)}
        path = os.path.join(self.MODEL_DIR, 'config.json')
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"[Config] Saved configuration to {path}")

# Setup môi trường ngay khi chạy
Config.setup_gpu()

# Kiểm tra đường dẫn in ra có đúng không
print(f"📂 PROJECT ROOT: {Config.BASE_DIR}")
print(f"📂 DATA FILE:    {Config.DATA_PATH}")

os.makedirs(Config.MODEL_DIR, exist_ok=True)

# Kiểm tra file dữ liệu
if not os.path.exists(Config.DATA_PATH):
    print(f"❌ Error: Vẫn không tìm thấy file tại {Config.DATA_PATH}")
    exit()
else:
    print(f"✅ Tìm thấy dữ liệu! Bắt đầu training...")

# ==========================================
# 2. DATASET & DATAMODULE
# ==========================================
class CAFA6Dataset:
    def __init__(self, data_path, max_len=512, num_classes=1500):
        self.data_path = data_path
        self.max_len = max_len
        self.num_classes = num_classes
        self.AA_MAP = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
        }
        self.top_terms = []
        self.term_to_idx = {}
        
        self.df = self.load_data()
        self.X, self.Y = self.process_data()

    def load_data(self):
        print(f"[Dataset] Loading .pkl file...")
        return pd.read_pickle(self.data_path)

    def process_data(self):
        print(f"[Dataset] Processing {len(self.df)} samples...")
        
        # 1. Tokenization
        X_list = [[self.AA_MAP.get(aa, 0) for aa in seq] for seq in self.df['sequence']]
        X = pad_sequences(X_list, maxlen=self.max_len, padding='post', truncating='post')
        
        # 2. Label Encoding
        all_terms = [t for sublist in self.df['term'] for t in sublist]
        self.top_terms = [t[0] for t in Counter(all_terms).most_common(self.num_classes)]
        self.term_to_idx = {t: i for i, t in enumerate(self.top_terms)}
        
        Y = np.zeros((len(self.df), self.num_classes), dtype='float32')
        for i, terms in enumerate(self.df['term']):
            for t in terms:
                if t in self.term_to_idx:
                    Y[i, self.term_to_idx[t]] = 1.0
                    
        print(f"[Dataset] Processed X shape: {X.shape}, Y shape: {Y.shape}")
        return X, Y
    
    def save_labels_map(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.top_terms, f)
        print(f"[Dataset] Saved labels map to {path}")

class CAFA6DataModule:
    def __init__(self, dataset, batch_size=64, test_size=0.1):
        self.X = dataset.X
        self.Y = dataset.Y
        self.batch_size = batch_size
        self.test_size = test_size
        self.X_val = None
        self.y_val = None
        self.train_ds = None
        self.val_ds = None

    def setup(self):
        print("[DataModule] Splitting Train/Val...")
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.Y, test_size=self.test_size, random_state=42
        )
        self.X_val, self.y_val = X_val, y_val
        
        print(f"   -> Train samples: {len(X_train)}")
        print(f"   -> Val samples:   {len(X_val)}")
        
        # Convert to tf.data.Dataset
        self.train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(1024)\
            .batch(self.batch_size)\
            .prefetch(tf.data.AUTOTUNE)
            
        self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
            .batch(self.batch_size)\
            .prefetch(tf.data.AUTOTUNE)

# ==========================================
# 3. MODEL ARCHITECTURE
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

class ProteinTransformerModel:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self):
        inputs = layers.Input(shape=(self.config.MAX_SEQ_LEN,))
        
        # Architecture
        x = TokenAndPositionEmbedding(self.config.MAX_SEQ_LEN, self.config.VOCAB_SIZE, self.config.EMBED_DIM)(inputs)
        x = TransformerBlock(self.config.EMBED_DIM, self.config.NUM_HEADS, self.config.FF_DIM)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(self.config.NUM_CLASSES, activation="sigmoid")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name="ProteinTransformer")
        
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
# 4. UTILS (VISUALIZATION)
# ==========================================
def plot_history(history, save_dir):
    # Save CSV
    pd.DataFrame(history.history).to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax[0].plot(history.history['loss'], label='Train Loss', color='blue')
    ax[0].plot(history.history['val_loss'], label='Val Loss', color='orange')
    ax[0].set_title('Loss (Binary Crossentropy)')
    ax[0].legend()
    ax[0].grid(True)
    
    # AUC
    ax[1].plot(history.history['auc'], label='Train AUC', color='green')
    ax[1].plot(history.history['val_auc'], label='Val AUC', color='red')
    ax[1].set_title('AUC Score')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_plot.png'))
    plt.show()

def predict_sample(model_wrapper, dm, dataset, num_samples=3):
    print("\n=== SAMPLE PREDICTION ===")
    X_sample = dm.X_val[:num_samples]
    y_true = dm.y_val[:num_samples]
    
    preds = model_wrapper.model.predict(X_sample)
    
    for i in range(num_samples):
        print(f"\nSample #{i+1}:")
        
        # Get True Labels
        true_idxs = np.where(y_true[i] == 1)[0]
        true_terms = [dataset.top_terms[k] for k in true_idxs]
        
        # Get Pred Labels (Threshold > 0.2)
        pred_idxs = np.where(preds[i] > 0.2)[0]
        pred_terms = [dataset.top_terms[k] for k in pred_idxs]
        
        print(f"  Target: {true_terms[:5]} ... (Total: {len(true_terms)})")
        print(f"  Predict: {pred_terms[:5]} ... (Total: {len(pred_terms)})")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    cfg = Config()
    cfg.save_to_json() # Lưu cấu hình
    
    # 1. Load Data
    dataset = CAFA6Dataset(cfg.DATA_PATH, cfg.MAX_SEQ_LEN, cfg.NUM_CLASSES)
    # Lưu Label Map để dùng cho Inference
    dataset.save_labels_map(os.path.join(cfg.MODEL_DIR, 'labels_map.pkl'))
    
    # 2. Data Module
    dm = CAFA6DataModule(dataset, cfg.BATCH_SIZE)
    dm.setup()
    
    # 3. Callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg.MODEL_DIR, 'best_model.keras'),
            monitor='val_auc', mode='max', save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=cfg.EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_auc', mode='max', factor=0.5, patience=cfg.REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1
        ),
        callbacks.CSVLogger(os.path.join(cfg.MODEL_DIR, 'training_log.csv'))
    ]
    
    # 4. Train
    # Cưỡng chế chạy trên GPU
    with tf.device('/GPU:0'):
        transformer = ProteinTransformerModel(cfg)
        transformer.model.summary()
        
        history = transformer.train(dm, callbacks_list)
        
        # Lưu Final Model
        transformer.model.save(os.path.join(cfg.MODEL_DIR, 'final_model.keras'))
        
    # 5. Evaluate
    plot_history(history, cfg.MODEL_DIR)
    
    print("\n=== FINAL EVALUATION ON VAL SET ===")
    val_metrics = transformer.model.evaluate(dm.val_ds)
    print(f"Loss: {val_metrics[0]:.4f} | Accuracy: {val_metrics[1]:.4f} | AUC: {val_metrics[2]:.4f}")
    
    predict_sample(transformer, dm, dataset)