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
# 1. CONFIGURATION
# ==========================================
class Config:
    # Project Root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    # Input Data: Tái sử dụng data từ ver3 (đỡ phải copy file nặng)
    # Nếu bạn đã move file sang ver4 thì sửa lại path này
    DATA_PATH = os.path.join(BASE_DIR, 'models', 'ver3', 'processed_data.pkl')
    
    # Output Model: Lưu kết quả vào ver4
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ver4')
    
    # Model Params
    MAX_SEQ_LEN = 512
    NUM_CLASSES = 1500
    VOCAB_SIZE  = 21
    TAX_EMBED_DIM = 16
    
    # Training Params
    BATCH_SIZE  = 64
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
        config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)}
        path = os.path.join(self.MODEL_DIR, 'config.json')
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"[Config] Saved configuration to {path}")

# Init Environment
Config.setup_gpu()
os.makedirs(Config.MODEL_DIR, exist_ok=True)

# Sanity Check
if not os.path.exists(Config.DATA_PATH):
    print(f"❌ Error: Không tìm thấy data tại {Config.DATA_PATH}")
    print("👉 Hãy copy file processed_data.pkl vào thư mục models/ver3 hoặc sửa DATA_PATH trong code.")
    exit()

# ==========================================
# 2. CUSTOM LOSS: ASYMMETRIC LOSS (ASL)
# ==========================================
class AsymmetricLoss(tf.keras.losses.Loss):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(y_pred, self.eps, 1 - self.eps)

        # Calculating Probabilities
        x_sigmoid = y_pred
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = tf.clip_by_value(xs_neg + self.clip, 0, 1)

        # Basic Cross Entropy
        los_pos = y_true * tf.math.log(xs_pos)
        los_neg = (1 - y_true) * tf.math.log(xs_neg)
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y_true
            pt1 = xs_neg * (1 - y_true)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y_true + self.gamma_neg * (1 - y_true)
            one_sided_w = tf.math.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -tf.reduce_sum(loss, axis=-1)

# ==========================================
# 3. DATASET
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
        self.tax_to_idx = {}
        self.num_taxons = 0
        
        self.df = self.load_data()
        self.X_seq, self.X_tax, self.Y = self.process_data()

    def load_data(self):
        print(f"[Dataset] Loading .pkl from {self.data_path}...")
        return pd.read_pickle(self.data_path)

    def process_data(self):
        print(f"[Dataset] Processing {len(self.df)} samples...")
        
        # 1. Sequence
        X_seq_list = [[self.AA_MAP.get(aa, 0) for aa in seq] for seq in self.df['sequence']]
        X_seq = pad_sequences(X_seq_list, maxlen=self.max_len, padding='post', truncating='post')
        
        # 2. Taxonomy
        unique_taxons = self.df['taxonomyID'].unique()
        self.tax_to_idx = {tax: i+1 for i, tax in enumerate(unique_taxons)}
        self.num_taxons = len(unique_taxons) + 1
        X_tax = np.array([self.tax_to_idx.get(t, 0) for t in self.df['taxonomyID']])
        
        # 3. Labels
        all_terms = [t for sublist in self.df['term'] for t in sublist]
        self.top_terms = [t[0] for t in Counter(all_terms).most_common(self.num_classes)]
        self.term_to_idx = {t: i for i, t in enumerate(self.top_terms)}
        
        Y = np.zeros((len(self.df), self.num_classes), dtype='float32')
        for i, terms in enumerate(self.df['term']):
            for t in terms:
                if t in self.term_to_idx:
                    Y[i, self.term_to_idx[t]] = 1.0
                    
        return X_seq, X_tax, Y
    
    def save_mappings(self, model_dir):
        with open(os.path.join(model_dir, 'labels_map.pkl'), 'wb') as f:
            pickle.dump(self.top_terms, f)
        with open(os.path.join(model_dir, 'tax_map.pkl'), 'wb') as f:
            pickle.dump(self.tax_to_idx, f)
        print(f"[Dataset] Mappings saved to {model_dir}")

# ==========================================
# 4. DATAMODULE
# ==========================================
class CAFA6DataModule:
    def __init__(self, dataset, batch_size=64, test_size=0.1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_size = test_size
        self.train_ds = None
        self.val_ds = None
        self.val_data = {}

    def setup(self):
        print("[DataModule] Splitting Train/Val...")
        X_seq_train, X_seq_val, X_tax_train, X_tax_val, y_train, y_val = train_test_split(
            self.dataset.X_seq, self.dataset.X_tax, self.dataset.Y, 
            test_size=self.test_size, random_state=42
        )
        
        self.val_data = {'seq': X_seq_val, 'tax': X_tax_val, 'y': y_val}
        
        train_inputs = {'seq_input': X_seq_train, 'tax_input': X_tax_train}
        val_inputs   = {'seq_input': X_seq_val,   'tax_input': X_tax_val}
        
        self.train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, y_train))\
            .shuffle(1024).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            
        self.val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, y_val))\
            .batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 5. MODEL (MULTI-INPUT + ASL LOSS)
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
        self.num_taxons = num_taxons
        self.model = self._build_model()
        
    def _build_model(self):
        # 1. Sequence Branch
        seq_input = layers.Input(shape=(self.config.MAX_SEQ_LEN,), name='seq_input')
        x1 = TokenAndPositionEmbedding(self.config.MAX_SEQ_LEN, self.config.VOCAB_SIZE, self.config.EMBED_DIM)(seq_input)
        x1 = TransformerBlock(self.config.EMBED_DIM, self.config.NUM_HEADS, self.config.FF_DIM)(x1)
        x1 = layers.GlobalAveragePooling1D()(x1)
        
        # 2. Taxonomy Branch
        tax_input = layers.Input(shape=(1,), name='tax_input')
        x2 = layers.Embedding(input_dim=self.num_taxons, output_dim=self.config.TAX_EMBED_DIM)(tax_input)
        x2 = layers.Flatten()(x2)
        
        # 3. Merge
        x = layers.Concatenate()([x1, x2])
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(self.config.NUM_CLASSES, activation="sigmoid")(x)
        
        model = models.Model(inputs=[seq_input, tax_input], outputs=outputs, name="ProteinMultiInput_ASL")
        
        # --- COMPILE WITH ASYMMETRIC LOSS ---
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss=AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05), # ASL Settings
            metrics=["binary_accuracy", tf.keras.metrics.AUC(multi_label=True, name='auc')]
        )
        return model

    def train(self, data_module, callbacks_list):
        print(f"\n[Model] Training with ASL ({self.config.EPOCHS} Epochs)...")
        return self.model.fit(
            data_module.train_ds,
            validation_data=data_module.val_ds,
            epochs=self.config.EPOCHS,
            callbacks=callbacks_list,
            verbose=1
        )

# ==========================================
# 6. EXECUTION
# ==========================================
def plot_history(history, save_dir):
    pd.DataFrame(history.history).to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title('Asymmetric Loss')
    ax[0].legend(); ax[0].grid(True)
    
    ax[1].plot(history.history['auc'], label='Train AUC')
    ax[1].plot(history.history['val_auc'], label='Val AUC')
    ax[1].set_title('AUC')
    ax[1].legend(); ax[1].grid(True)
    
    plt.savefig(os.path.join(save_dir, 'training_plot.png'))
    plt.show()

if __name__ == "__main__":
    cfg = Config()
    
    # 1. Dataset
    dataset = CAFA6Dataset(cfg.DATA_PATH, cfg.MAX_SEQ_LEN, cfg.NUM_CLASSES)
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
    
    # 4. Train
    num_taxons = dataset.num_taxons
    print(f"\n[Model] Initializing for {num_taxons} species with Asymmetric Loss...")
    
    with tf.device('/GPU:0'):
        model_wrapper = ProteinMultiInputModel(cfg, num_taxons)
        model_wrapper.model.summary()
        
        history = model_wrapper.train(dm, callbacks_list)
        model_wrapper.model.save(os.path.join(cfg.MODEL_DIR, 'final_model.keras'))
        
    # 5. Report
    cfg.save_to_json()
    plot_history(history, cfg.MODEL_DIR)
    
    print("\n=== FINAL EVALUATION ===")
    val_metrics = model_wrapper.model.evaluate(dm.val_ds)
    print(f"Loss: {val_metrics[0]:.4f} | Accuracy: {val_metrics[1]:.4f} | AUC: {val_metrics[2]:.4f}")