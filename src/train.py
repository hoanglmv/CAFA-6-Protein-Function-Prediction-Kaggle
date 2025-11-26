import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from IPython.display import display

# ==========================================
# 1. CONFIGURATION (CẤU HÌNH)
# ==========================================
class Config:
    # Đường dẫn
    BASE_DIR = r'/home/myvh/hoang/CAFA-6-Protein-Function-Prediction-Kaggle'
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.pkl')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # Tham số mô hình
    MAX_SEQ_LEN = 512       # Độ dài chuỗi
    NUM_CLASSES = 1500      # Số lượng nhãn GO phổ biến nhất
    VOCAB_SIZE  = 21        # 20 Axit amin + 1 padding
    
    # Tham số huấn luyện
    BATCH_SIZE  = 64        # Tối ưu cho RTX 3060 12GB
    EPOCHS      = 15
    LEARNING_RATE = 1e-3
    EMBED_DIM   = 64
    NUM_HEADS   = 4
    FF_DIM      = 128
    
    # Setup GPU
    @staticmethod
    def setup_gpu():
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

Config.setup_gpu()
os.makedirs(Config.MODEL_DIR, exist_ok=True)

# ==========================================
# 2. CLASS DATASET (XỬ LÝ DỮ LIỆU THÔ)
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
        
        # Load dữ liệu ngay khi khởi tạo
        self.df = self.load_data()
        self.X, self.Y = self.process_data()

    def load_data(self):
        print(f"[Dataset] Loading data from {self.data_path}...")
        df = pd.read_pickle(self.data_path)
        print(f"[Dataset] Loaded {len(df)} samples.")
        return df

    def process_data(self):
        print("[Dataset] Processing Sequences (Tokenization)...")
        # 1. Tokenize Sequence
        X_list = [[self.AA_MAP.get(aa, 0) for aa in seq] for seq in self.df['sequence']]
        X = pad_sequences(X_list, maxlen=self.max_len, padding='post', truncating='post')
        
        print("[Dataset] Processing Labels (One-hot Encoding)...")
        # 2. Encode Labels
        all_terms = [t for sublist in self.df['term'] for t in sublist]
        self.top_terms = [t[0] for t in Counter(all_terms).most_common(self.num_classes)]
        self.term_to_idx = {t: i for i, t in enumerate(self.top_terms)}
        
        Y = np.zeros((len(self.df), self.num_classes), dtype='float32')
        for i, terms in enumerate(self.df['term']):
            for t in terms:
                if t in self.term_to_idx:
                    Y[i, self.term_to_idx[t]] = 1.0
        
        return X, Y
    
    def save_labels_map(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.top_terms, f)
        print(f"[Dataset] Saved label map to {path}")

# ==========================================
# 3. CLASS DATAMODULE (CHIA TẬP & BATCHING)
# ==========================================
class CAFA6DataModule:
    def __init__(self, dataset, batch_size=64, test_size=0.1):
        self.X = dataset.X
        self.Y = dataset.Y
        self.batch_size = batch_size
        self.test_size = test_size
        self.train_ds = None
        self.val_ds = None
        self.X_val = None # Lưu lại để dùng cho việc Predict thủ công
        self.y_val = None

    def setup(self):
        print("[DataModule] Splitting Train/Validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.Y, test_size=self.test_size, random_state=42
        )
        self.X_val = X_val
        self.y_val = y_val
        
        print(f"[DataModule] Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        
        # Tạo tf.data.Dataset (Tăng tốc độ train trên GPU)
        self.train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(buffer_size=1024)\
            .batch(self.batch_size)\
            .prefetch(tf.data.AUTOTUNE)
            
        self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
            .batch(self.batch_size)\
            .prefetch(tf.data.AUTOTUNE)

# ==========================================
# 4. CLASS MODEL (TRANSFORMER ARCHITECTURE)
# ==========================================
# --- Custom Layers ---
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

# --- Main Model Class ---
class ProteinTransformerModel:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()
    
    def build_model(self):
        inputs = layers.Input(shape=(self.config.MAX_SEQ_LEN,))
        
        # 1. Embedding
        x = TokenAndPositionEmbedding(self.config.MAX_SEQ_LEN, self.config.VOCAB_SIZE, self.config.EMBED_DIM)(inputs)
        
        # 2. Transformer Block
        x = TransformerBlock(self.config.EMBED_DIM, self.config.NUM_HEADS, self.config.FF_DIM)(x)
        
        # 3. Head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(self.config.NUM_CLASSES, activation="sigmoid")(inputs=x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name="ProteinTransformer")
        
        # Compile Model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=[
                "binary_accuracy", 
                tf.keras.metrics.AUC(multi_label=True, name='auc')
            ]
        )
        return model

    def train(self, data_module):
        print(f"\n[Model] Starting training for {self.config.EPOCHS} epochs...")
        history = self.model.fit(
            data_module.train_ds,
            validation_data=data_module.val_ds,
            epochs=self.config.EPOCHS,
            verbose=1
        )
        return history

    def save(self, path):
        self.model.save(path)
        print(f"[Model] Saved model to {path}")

# ==========================================
# 5. EXECUTION & VISUALIZATION
# ==========================================
def plot_history(history, save_dir):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(save_dir, 'history.csv'), index=False)
    
    plt.figure(figsize=(14, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Binary Crossentropy)')
    plt.legend()
    plt.grid(True)
    
    # Plot AUC
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('Training & Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'training_plot.png'))
    plt.show()

def evaluate_sample(model_wrapper, data_module, dataset, num_samples=3):
    print("\n=== DỰ ĐOÁN THỬ (SAMPLE PREDICTION) ===")
    
    # Lấy mẫu từ tập Validation
    X_sample = data_module.X_val[:num_samples]
    y_true_sample = data_module.y_val[:num_samples]
    
    # Dự đoán
    y_pred_probs = model_wrapper.model.predict(X_sample)
    
    for i in range(num_samples):
        print(f"\nSample #{i+1}:")
        
        # Lấy các nhãn thật (Ground Truth)
        true_indices = np.where(y_true_sample[i] == 1)[0]
        true_terms = [dataset.top_terms[idx] for idx in true_indices]
        
        # Lấy các nhãn dự đoán (ngưỡng > 0.2)
        pred_indices = np.where(y_pred_probs[i] > 0.2)[0] 
        pred_terms = [dataset.top_terms[idx] for idx in pred_indices]
        
        print(f" - True Terms ({len(true_terms)}): {true_terms[:5]} ...")
        print(f" - Pred Terms ({len(pred_terms)}): {pred_terms[:5]} ... (Score > 0.2)")

# --- RUNNING THE PIPELINE ---
if __name__ == "__main__":
    # 1. Init Config
    cfg = Config()
    
    # 2. Prepare Dataset
    dataset = CAFA6Dataset(cfg.DATA_PATH, cfg.MAX_SEQ_LEN, cfg.NUM_CLASSES)
    # Lưu label map để sau này dùng cho file submission
    dataset.save_labels_map(os.path.join(cfg.MODEL_DIR, 'labels_map.pkl'))
    
    # 3. Prepare Data Module
    dm = CAFA6DataModule(dataset, cfg.BATCH_SIZE)
    dm.setup()
    
    # 4. Build & Train Model
    # Dùng context manager để đảm bảo model nằm trên GPU
    with tf.device('/GPU:0'):
        transformer_model = ProteinTransformerModel(cfg)
        transformer_model.model.summary()
        
        # Train
        history = transformer_model.train(dm)
        
        # Save
        transformer_model.save(os.path.join(cfg.MODEL_DIR, 'transformer_model.keras'))
    
    # 5. Visualize & Evaluate
    plot_history(history, cfg.MODEL_DIR)
    
    # Tính toán Accuracy cuối cùng trên tập Val
    print("\n=== ĐÁNH GIÁ MÔ HÌNH (FINAL EVALUATION) ===")
    val_loss, val_acc, val_auc = transformer_model.model.evaluate(dm.val_ds)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy (Binary): {val_acc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    
    # Dự đoán thử
    evaluate_sample(transformer_model, dm, dataset)