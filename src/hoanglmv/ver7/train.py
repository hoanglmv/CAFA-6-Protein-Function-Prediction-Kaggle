import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, initializers

# ==========================================
# 1. CẤU HÌNH
# ==========================================
class Config:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed2')
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ver7')
    
    TRAIN_FILE = os.path.join(DATA_DIR, 'train.parquet')
    
    # Onto2Vec Params
    ONTO_EMBED_DIM = 200  # Giả sử Onto2Vec có chiều là 200
    USE_ONTO2VEC = True   # Bật chế độ này
    
    BATCH_SIZE = 256
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 1500
    
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
# 2. CUSTOM LOSS (Giữ nguyên)
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

# ==========================================
# 3. GIẢ LẬP LOAD ONTO2VEC
# ==========================================
def load_onto2vec_embeddings(idx_to_term, embed_dim):
    """
    Trong thực tế, bạn sẽ load file pre-trained Onto2Vec (word2vec format).
    Ở đây tôi tạo ngẫu nhiên để demo cấu trúc code.
    """
    print("🔄 Generating Mock Onto2Vec Embeddings (Thay thế bằng file thật khi có)...")
    
    # Ma trận (Số Class, Số chiều Onto2Vec)
    # Hàng thứ i tương ứng với vector của class i
    embedding_matrix = np.random.normal(0, 0.1, (Config.NUM_CLASSES, embed_dim)).astype(np.float32)
    
    # Nếu bạn có file thật, code sẽ như sau:
    # model_onto = KeyedVectors.load_word2vec_format('onto2vec.bin', binary=True)
    # for idx, term_str in idx_to_term.items():
    #     if term_str in model_onto:
    #         embedding_matrix[idx] = model_onto[term_str]
            
    return embedding_matrix

# ==========================================
# 4. LOAD DATA (Giữ nguyên phần xử lý)
# ==========================================
def load_and_process_data(filepath):
    # ... (Giữ nguyên code load data của bạn ở các bước trước) ...
    # Để tiết kiệm không gian hiển thị, tôi tóm tắt lại
    print(f"📖 Đang đọc file: {filepath}")
    df = pd.read_parquet(filepath)
    X_emb = np.stack(df['embedding'].values)
    X_tax = np.stack(df['superkingdom'].values)
    
    # Map Labels
    idx_to_term = {}
    for terms, ids in zip(df['go_terms'], df['go_terms_id']):
        if len(idx_to_term) >= Config.NUM_CLASSES: break
        for term_str, term_id in zip(terms, ids):
            idx_to_term[term_id] = term_str
            
    # Matrix Y
    Y = np.zeros((len(df), Config.NUM_CLASSES), dtype='float32')
    for i, ids in enumerate(df['go_terms_id']):
        for term_id in ids:
            if term_id < Config.NUM_CLASSES:
                Y[i, term_id] = 1.0
                
    return X_emb, X_tax, Y, idx_to_term

# ==========================================
# 5. MÔ HÌNH TÍCH HỢP ONTO2VEC
# ==========================================
def build_model_with_onto2vec(emb_dim, tax_dim, go_embedding_matrix):
    
    # --- Input Processing ---
    input_emb = layers.Input(shape=(emb_dim,), name='input_embedding')
    x1 = layers.Dense(512)(input_emb)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Dropout(0.3)(x1)
    
    input_tax = layers.Input(shape=(tax_dim,), name='input_taxonomy')
    x2 = layers.Dense(32, activation='relu')(input_tax) # Tăng nhẹ tax dense
    
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # --- KEY CHANGE: LABEL PROJECTION ---
    
    # Bước 1: Project đặc trưng Protein về cùng không gian với Onto2Vec (200 dim)
    # Lớp này học cách biến đổi Protein -> Semantic Space
    protein_semantic_vector = layers.Dense(Config.ONTO_EMBED_DIM, activation=None, name='protein_projection')(x)
    protein_semantic_vector = layers.BatchNormalization()(protein_semantic_vector) # Normalize để dot product ổn định
    
    # Bước 2: Tạo lớp chứa Pre-trained GO Embeddings
    # Chúng ta dùng lớp này như một hằng số (không train lại) hoặc fine-tune nhẹ
    # Shape: (200, 1500) -> Transpose của ma trận (1500, 200)
    go_emb_tensor = tf.constant(go_embedding_matrix.T, dtype=tf.float32)
    
    # Bước 3: Tính Dot Product (Độ tương đồng)
    # Output shape: (Batch, 200) x (200, 1500) -> (Batch, 1500)
    # Đây chính là logits cho mỗi class
    
    # Sử dụng Lambda layer để thực hiện phép nhân ma trận với hằng số
    logits = layers.Lambda(lambda v: tf.matmul(v, go_emb_tensor), name='cosine_similarity')(protein_semantic_vector)
    
    # Bước 4: Sigmoid để ra xác suất
    output = layers.Activation('sigmoid', name='output')(logits)
    
    model = models.Model(inputs=[input_emb, input_tax], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=Config.LEARNING_RATE),
        loss=AsymmetricLoss(),
        metrics=[tf.keras.metrics.AUC(name='auc'), 'binary_accuracy']
    )
    return model

if __name__ == "__main__":
    X_emb, X_tax, Y, idx_to_term = load_and_process_data(Config.TRAIN_FILE)
    
    # 1. Tạo/Load Onto2Vec Matrix
    # Ở bước này, nếu bạn có file Onto2Vec thật, độ chính xác sẽ tăng mạnh
    go_vectors = load_onto2vec_embeddings(idx_to_term, Config.ONTO_EMBED_DIM)
    
    # Split
    X_emb_train, X_emb_val, X_tax_train, X_tax_val, Y_train, Y_val = train_test_split(
        X_emb, X_tax, Y, test_size=0.1, random_state=42
    )
    
    # Build Model đặc biệt
    print(f"🏗️ Building Model with Onto2Vec Integration...")
    model = build_model_with_onto2vec(X_emb.shape[1], X_tax.shape[1], go_vectors)
    model.summary()
    
    # Callbacks
    cbs = [
        callbacks.ModelCheckpoint(os.path.join(Config.MODEL_DIR, 'best_model_onto.keras'), 
                                  save_best_only=True, monitor='val_auc', mode='max'),
        callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3)
    ]
    
    # Train
    model.fit(
        x={'input_embedding': X_emb_train, 'input_taxonomy': X_tax_train},
        y=Y_train,
        validation_data=({'input_embedding': X_emb_val, 'input_taxonomy': X_tax_val}, Y_val),
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        callbacks=cbs
    )