import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks

# ==========================================
# 1. CẤU HÌNH
# ==========================================
class Config:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed2')
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ver6')
    
    TRAIN_FILE = os.path.join(DATA_DIR, 'train.parquet')
    
    # Hyperparameters
    BATCH_SIZE = 64 # Có thể tăng lên 128/256 nếu RAM nhiều
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 1500 # Số lượng nhãn GO
    
    @staticmethod
    def setup_gpu():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU Activated: {gpus[0].name}")
            except: pass

Config.setup_gpu()
os.makedirs(Config.MODEL_DIR, exist_ok=True)

# ==========================================
# 2. CUSTOM LOSS (ASYMMETRIC LOSS)
# ==========================================
class AsymmetricLoss(tf.keras.losses.Loss):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, self.eps, 1 - self.eps)
        xs_pos = y_pred
        xs_neg = 1 - y_pred
        if self.clip > 0: xs_neg = tf.clip_by_value(xs_neg + self.clip, 0, 1)
        loss = (y_true * tf.math.log(xs_pos)) + ((1 - y_true) * tf.math.log(xs_neg))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = (xs_pos * y_true) + (xs_neg * (1 - y_true))
            loss *= tf.math.pow(1 - pt, self.gamma_pos * y_true + self.gamma_neg * (1 - y_true))
        return -tf.reduce_sum(loss, axis=-1)
    
    def get_config(self):
        return {'gamma_neg': self.gamma_neg, 'gamma_pos': self.gamma_pos, 'clip': self.clip, 'eps': self.eps}

# ==========================================
# 3. HÀM XỬ LÝ DỮ LIỆU
# ==========================================
def load_and_process_data(filepath):
    print(f"📖 Đang đọc file: {filepath}")
    df = pd.read_parquet(filepath)
    
    # 1. Chuyển đổi Embedding (List -> Numpy Array)
    # df['embedding'] đang là cột chứa các list, cần stack thành ma trận 2D
    print("   -> Processing Embeddings...")
    X_emb = np.stack(df['embedding'].values)
    
    # 2. Chuyển đổi Superkingdom (One-hot)
    print("   -> Processing Taxonomy Groups...")
    X_tax = np.stack(df['superkingdom'].values)
    
    # 3. Xử lý Nhãn (Labels) & Tạo Mapping
    print("   -> Processing Labels & Creating Map...")
    
    # Tạo mapping: index -> GO Term String
    # Dựa vào cột 'go_terms_id' (index) và 'go_terms' (string)
    # Ta cần lấy mẫu để xây dựng lại từ điển này
    idx_to_term = {}
    
    # Duyệt qua một số dòng để map lại (hoặc duyệt hết nếu cần chính xác 100%)
    # Giả định: go_terms_id và go_terms đồng bộ thứ tự
    for terms, ids in zip(df['go_terms'], df['go_terms_id']):
        if len(idx_to_term) >= Config.NUM_CLASSES: break
        for term_str, term_id in zip(terms, ids):
            idx_to_term[term_id] = term_str
            
    # Tạo ma trận Y (Multi-hot)
    Y = np.zeros((len(df), Config.NUM_CLASSES), dtype='float32')
    for i, ids in enumerate(df['go_terms_id']):
        for term_id in ids:
            if term_id < Config.NUM_CLASSES:
                Y[i, term_id] = 1.0
                
    return X_emb, X_tax, Y, idx_to_term

# ==========================================
# 4. XÂY DỰNG MODEL
# ==========================================
def build_model(emb_dim, tax_dim):
    # Nhánh 1: Embedding (ESM2)
    input_emb = layers.Input(shape=(emb_dim,), name='input_embedding')
    x1 = layers.Dense(512)(input_emb)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Dropout(0.3)(x1)
    
    # Nhánh 2: Taxonomy (One-hot 4 chiều)
    input_tax = layers.Input(shape=(tax_dim,), name='input_taxonomy')
    x2 = layers.Dense(16, activation='relu')(input_tax)
    
    # Kết hợp
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    output = layers.Dense(Config.NUM_CLASSES, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=[input_emb, input_tax], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss=AsymmetricLoss(),
        metrics=['binary_accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    # Load Data
    X_emb, X_tax, Y, idx_to_term = load_and_process_data(Config.TRAIN_FILE)
    
    # Lưu lại file map để dùng cho test.py
    map_path = os.path.join(Config.MODEL_DIR, 'idx_to_term.pkl')
    with open(map_path, 'wb') as f:
        pickle.dump(idx_to_term, f)
    print(f"✅ Saved label mapping to {map_path}")
    
    # Split Train/Val
    print("✂️ Splitting Train/Val...")
    X_emb_train, X_emb_val, X_tax_train, X_tax_val, Y_train, Y_val = train_test_split(
        X_emb, X_tax, Y, test_size=0.1, random_state=42
    )
    
    # Build Model
    emb_dim = X_emb.shape[1]
    tax_dim = X_tax.shape[1]
    print(f"🏗️ Building Model (Emb Dim: {emb_dim}, Tax Dim: {tax_dim})...")
    
    model = build_model(emb_dim, tax_dim)
    model.summary()
    
    # Callbacks
    cbs = [
        callbacks.ModelCheckpoint(os.path.join(Config.MODEL_DIR, 'best_model.keras'), 
                                  save_best_only=True, monitor='val_auc', mode='max', verbose=1),
        callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, verbose=1)
    ]
    
    # Train
    print("🚀 Start Training...")
    history = model.fit(
        x={'input_embedding': X_emb_train, 'input_taxonomy': X_tax_train},
        y=Y_train,
        validation_data=({'input_embedding': X_emb_val, 'input_taxonomy': X_tax_val}, Y_val),
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        callbacks=cbs
    )
    
    print("✅ Training Finished!")