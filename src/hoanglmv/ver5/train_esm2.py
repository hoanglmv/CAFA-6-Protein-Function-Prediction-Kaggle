import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from collections import Counter

# ==========================================
# CẤU HÌNH
# ==========================================
class Config:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'esm2_ver1')
    
    # Input Data
    TRAIN_PKL = os.path.join(BASE_DIR, 'models', 'ver4', 'processed_data.pkl')
    TRAIN_EMB = os.path.join(MODEL_DIR, 'train_embeddings.npy')
    
    # Params
    ESM_DIM = 320 # Nếu dùng esm2_t6_8M. Nếu dùng t12 thì là 480, t33 là 1280
    NUM_CLASSES = 1500
    TAX_EMBED_DIM = 16
    
    BATCH_SIZE = 128 # ESM vector nhẹ, tăng batch lên
    EPOCHS = 50
    LEARNING_RATE = 5e-4

Config.setup_gpu = lambda: tf.config.list_physical_devices('GPU') and print("✅ GPU Ready")

# ==========================================
# ASYMMETRIC LOSS (Copy lại)
# ==========================================
class AsymmetricLoss(tf.keras.losses.Loss):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, self.eps, 1 - self.eps)
        xs_pos = y_pred; xs_neg = 1 - y_pred
        if self.clip > 0: xs_neg = tf.clip_by_value(xs_neg + self.clip, 0, 1)
        loss = (y_true * tf.math.log(xs_pos)) + ((1 - y_true) * tf.math.log(xs_neg))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = (xs_pos * y_true) + (xs_neg * (1 - y_true))
            loss *= tf.math.pow(1 - pt, self.gamma_pos * y_true + self.gamma_neg * (1 - y_true))
        return -tf.reduce_sum(loss, axis=-1)
    def get_config(self): return {'gamma_neg': self.gamma_neg, 'gamma_pos': self.gamma_pos, 'clip': self.clip, 'eps': self.eps}

# ==========================================
# DATA & MODEL
# ==========================================
def load_data():
    print("Loading Dataframes & Embeddings...")
    df = pd.read_pickle(Config.TRAIN_PKL)
    X_emb = np.load(Config.TRAIN_EMB) # Vector từ ESM-2
    
    # Process Taxonomy
    taxs = df['taxonomyID'].unique()
    tax_to_idx = {t: i+1 for i, t in enumerate(taxs)}
    num_taxons = len(taxs) + 1
    X_tax = np.array([tax_to_idx.get(t, 0) for t in df['taxonomyID']])
    
    # Process Labels
    all_terms = [t for sub in df['term'] for t in sub]
    top_terms = [t[0] for t in Counter(all_terms).most_common(Config.NUM_CLASSES)]
    term_to_idx = {t: i for i, t in enumerate(top_terms)}
    Y = np.zeros((len(df), Config.NUM_CLASSES), dtype='float32')
    for i, terms in enumerate(df['term']):
        for t in terms:
            if t in term_to_idx: Y[i, term_to_idx[t]] = 1.0
            
    # Save maps
    with open(os.path.join(Config.MODEL_DIR, 'labels_map.pkl'), 'wb') as f: pickle.dump(top_terms, f)
    with open(os.path.join(Config.MODEL_DIR, 'tax_map.pkl'), 'wb') as f: pickle.dump(tax_to_idx, f)
    
    return X_emb, X_tax, Y, num_taxons

def build_esm_model(num_taxons):
    # Input 1: ESM Embedding Vector
    emb_in = layers.Input(shape=(Config.ESM_DIM,), name='esm_input')
    x1 = layers.Dense(512, activation='relu')(emb_in)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.3)(x1)
    
    # Input 2: Taxonomy
    tax_in = layers.Input(shape=(1,), name='tax_input')
    x2 = layers.Flatten()(layers.Embedding(num_taxons, Config.TAX_EMBED_DIM)(tax_in))
    
    # Merge
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    out = layers.Dense(Config.NUM_CLASSES, activation='sigmoid')(x)
    
    model = models.Model(inputs=[emb_in, tax_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE), 
                  loss=AsymmetricLoss(), metrics=['binary_accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

if __name__ == "__main__":
    Config.setup_gpu()
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    X_emb, X_tax, Y, num_taxons = load_data()
    
    # Split
    X_emb_tr, X_emb_val, X_tax_tr, X_tax_val, y_tr, y_val = train_test_split(X_emb, X_tax, Y, test_size=0.1, random_state=42)
    
    # TF Dataset
    train_ds = tf.data.Dataset.from_tensor_slices(({'esm_input': X_emb_tr, 'tax_input': X_tax_tr}, y_tr)).shuffle(1024).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(({'esm_input': X_emb_val, 'tax_input': X_tax_val}, y_val)).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Train
    model = build_esm_model(num_taxons)
    model.summary()
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=Config.EPOCHS, callbacks=[
        callbacks.ModelCheckpoint(os.path.join(Config.MODEL_DIR, 'best_model.keras'), monitor='val_auc', mode='max', save_best_only=True),
        callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3)
    ])
    print("✅ Training Finished!")