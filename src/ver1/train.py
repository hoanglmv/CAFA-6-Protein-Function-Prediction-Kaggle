import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from IPython.display import display

# ==========================================
# 1. CONFIGURATION (CẤU HÌNH)
# ==========================================
class Config:
    # Nếu file này nằm ở: src/ver1, lùi lên 2 cấp để tới project root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Input data: file processed_data.pkl (tôi đặt ở models/ver1 như bạn muốn)
    DATA_PATH = os.path.join(BASE_DIR, 'models', 'ver1', 'processed_data.pkl')

    # Output folder (nơi lưu model, history, labels_map, ảnh)
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ver1')

    # Model params
    MAX_SEQ_LEN = 512
    NUM_CLASSES = 1500
    VOCAB_SIZE = 21

    # Training params
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    EMBED_DIM = 64
    NUM_HEADS = 4
    FF_DIM = 128

    @staticmethod
    def setup_gpu():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for g in gpus:
                    tf.config.experimental.set_memory_growth(g, True)
                print(f"✅ GPU Activated: {gpus[0].name}")
            except RuntimeError as e:
                print("GPU setup error:", e)
        else:
            print("⚠️ Running on CPU")

# Prepare folders & GPU
Config.setup_gpu()
os.makedirs(Config.MODEL_DIR, exist_ok=True)

# Sanity check for data file
if not os.path.exists(Config.DATA_PATH):
    raise FileNotFoundError(f"Processed data not found: {Config.DATA_PATH}\n"
                            f"Put processed_data.pkl into {Config.MODEL_DIR} or update DATA_PATH.")

# ==========================================
# 2. DATASET LOADER
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
        print(f"[Dataset] Loading data from: {self.data_path}")
        df = pd.read_pickle(self.data_path)
        print(f"[Dataset] Loaded {len(df)} samples.")
        return df

    def process_data(self):
        print("[Dataset] Tokenizing sequences...")
        X_list = [[self.AA_MAP.get(aa, 0) for aa in seq] for seq in self.df['sequence']]
        X = pad_sequences(X_list, maxlen=self.max_len, padding='post', truncating='post')

        print("[Dataset] Encoding labels...")
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
        print(f"[Dataset] Saved label map → {path}")

# ==========================================
# 3. DATA MODULE (Tạo batch, chia tập)
# ==========================================
class CAFA6DataModule:
    def __init__(self, dataset, batch_size=64, test_size=0.1):
        self.X = dataset.X
        self.Y = dataset.Y
        self.batch_size = batch_size
        self.test_size = test_size
        self.train_ds = None
        self.val_ds = None
        self.X_val = None
        self.y_val = None

    def setup(self):
        print("[DataModule] Splitting train/val...")
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.Y, test_size=self.test_size, random_state=42
        )
        self.X_val = X_val
        self.y_val = y_val

        self.train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .shuffle(1024) \
            .batch(self.batch_size) \
            .prefetch(tf.data.AUTOTUNE)

        self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
            .batch(self.batch_size) \
            .prefetch(tf.data.AUTOTUNE)

# ==========================================
# 4. TRANSFORMER MODEL
# ==========================================
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
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

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class ProteinTransformerModel:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=(self.config.MAX_SEQ_LEN,))
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
            metrics=["binary_accuracy", tf.keras.metrics.AUC(multi_label=True, name="auc")]
        )
        return model

    def train(self, data_module, callbacks_list=None):
        print(f"\n[Model] Training for {self.config.EPOCHS} epochs...")
        return self.model.fit(
            data_module.train_ds,
            validation_data=data_module.val_ds,
            epochs=self.config.EPOCHS,
            verbose=1,
            callbacks=callbacks_list
        )

    def save(self, path):
        self.model.save(path)
        print(f"[Model] Saved → {path}")

# ==========================================
# 5. VISUALIZATION & HELPERS
# ==========================================
def plot_history(history, save_dir):
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(save_dir, "history.csv"), index=False)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('auc', []), label='train_auc')
    plt.plot(history.history.get('val_auc', []), label='val_auc')
    plt.title("AUC")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_plot.png"))
    plt.show()

def evaluate_sample(model_wrapper, data_module, dataset, num_samples=3):
    print("\n=== SAMPLE PREDICTIONS ===")
    X_sample = data_module.X_val[:num_samples]
    y_true_sample = data_module.y_val[:num_samples]

    y_pred_probs = model_wrapper.model.predict(X_sample)

    for i in range(num_samples):
        print(f"\nSample #{i+1}:")
        true_indices = np.where(y_true_sample[i] == 1)[0]
        true_terms = [dataset.top_terms[idx] for idx in true_indices]

        pred_indices = np.where(y_pred_probs[i] > 0.2)[0]
        pred_terms = [dataset.top_terms[idx] for idx in pred_indices]

        print(f" - True Terms ({len(true_terms)}): {true_terms[:5]} ...")
        print(f" - Pred Terms ({len(pred_terms)}): {pred_terms[:5]} ... (Score > 0.2)")

# ==========================================
# 6. MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    cfg = Config()

    # 1) Load dataset
    dataset = CAFA6Dataset(cfg.DATA_PATH, cfg.MAX_SEQ_LEN, cfg.NUM_CLASSES)

    # 2) Save label map into MODEL_DIR
    labels_path = os.path.join(cfg.MODEL_DIR, "labels_map.pkl")
    dataset.save_labels_map(labels_path)

    # 3) Prepare DataModule
    dm = CAFA6DataModule(dataset, cfg.BATCH_SIZE)
    dm.setup()

    # 4) Callbacks: checkpoint best model + early stopping + csv logger
    checkpoint_path = os.path.join(cfg.MODEL_DIR, "best_model.keras")
    final_model_path = os.path.join(cfg.MODEL_DIR, "transformer_model.keras")
    csv_log_path = os.path.join(cfg.MODEL_DIR, "training_log.csv")

    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    cb_early = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    cb_csv = callbacks.CSVLogger(csv_log_path)

    # 5) Build & Train
    try:
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        with tf.device(device):
            model_wrapper = ProteinTransformerModel(cfg)
            model_wrapper.model.summary()
            history = model_wrapper.train(dm, callbacks_list=[cb_checkpoint, cb_early, cb_csv])

            # Save final model (also keep checkpoint)
            model_wrapper.save(final_model_path)

    except Exception as e:
        # fallback train on default device if with tf.device fails
        print("Warning: training in device context failed:", e)
        model_wrapper = ProteinTransformerModel(cfg)
        history = model_wrapper.train(dm, callbacks_list=[cb_checkpoint, cb_early, cb_csv])
        model_wrapper.save(final_model_path)

    # 6) Visualize & evaluate
    plot_history(history, cfg.MODEL_DIR)

    print("\n=== FINAL EVALUATION ===")
    val_loss, val_acc, val_auc = model_wrapper.model.evaluate(dm.val_ds)
    print(f"VAL LOSS: {val_loss:.4f}")
    print(f"VAL ACC : {val_acc:.4f}")
    print(f"VAL AUC : {val_auc:.4f}")

    # 7) Quick sample predictions
    evaluate_sample(model_wrapper, dm, dataset)
