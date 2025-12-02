import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
from tensorflow.keras import models

# Config (Giống Train)
class Config:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'esm2_ver1')
    TEST_FASTA = os.path.join(BASE_DIR, 'data', 'Test', 'testsuperset.fasta')
    TEST_EMB = os.path.join(MODEL_DIR, 'test_embeddings.npy')
    TEST_IDS = os.path.join(MODEL_DIR, 'test_ids.pkl')
    
    TAX_CONVERSION = os.path.join(BASE_DIR, 'models', 'ver4', 'taxonomy_mapping.tsv') # Dùng lại file map của ver4
    SUBMISSION = os.path.join(BASE_DIR, 'submission_esm2.tsv')
    BATCH_SIZE = 128

# Custom Loss (để load model)
class AsymmetricLoss(tf.keras.losses.Loss):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, **kwargs): super().__init__(**kwargs)
    def call(self, y, p): return 0.0
    def get_config(self): return {}

def run_inference():
    print("🚀 Starting Inference with ESM-2...")
    
    # 1. Load Resources
    model = tf.keras.models.load_model(
        os.path.join(Config.MODEL_DIR, 'best_model.keras'), 
        custom_objects={'AsymmetricLoss': AsymmetricLoss}
    )
    with open(os.path.join(Config.MODEL_DIR, 'labels_map.pkl'), 'rb') as f: top_terms = pickle.load(f)
    with open(os.path.join(Config.MODEL_DIR, 'tax_map.pkl'), 'rb') as f: tax_idx_map = pickle.load(f)
    with open(Config.TEST_IDS, 'rb') as f: test_ids = pickle.load(f)
    
    # Load Tax Conversion Map
    tax_convert = {}
    if os.path.exists(Config.TAX_CONVERSION):
        df_map = pd.read_csv(Config.TAX_CONVERSION, sep='\t', dtype=str)
        tax_convert = dict(zip(df_map['Original_ID'], df_map['Mapped_ID']))
    
    # 2. Prepare Data
    # A. ESM Embeddings
    print("   -> Loading ESM Embeddings...")
    X_emb = np.load(Config.TEST_EMB)
    
    # B. Taxonomy IDs
    print("   -> Parsing Taxonomy from FASTA...")
    test_taxs = []
    for record in SeqIO.parse(Config.TEST_FASTA, "fasta"):
        parts = record.description.split()
        raw_tax = parts[1] if len(parts) >= 2 and parts[1].isdigit() else "0"
        mapped_tax = tax_convert.get(raw_tax, raw_tax)
        test_taxs.append(tax_idx_map.get(str(mapped_tax), 0))
    X_tax = np.array(test_taxs)
    
    # 3. Predict
    print(f"   -> Predicting {len(test_ids)} sequences...")
    with open(Config.SUBMISSION, 'w') as f:
        f.write("ObjectId\tGO-Term\tPrediction\n")
        
        for i in range(0, len(test_ids), Config.BATCH_SIZE):
            end = min(i + Config.BATCH_SIZE, len(test_ids))
            
            # Batch inputs
            batch_emb = X_emb[i:end]
            batch_tax = X_tax[i:end]
            
            preds = model.predict({'esm_input': batch_emb, 'tax_input': batch_tax}, verbose=0)
            
            # Write
            for j, pid in enumerate(test_ids[i:end]):
                probs = preds[j]
                top_k = np.argsort(probs)[-60:][::-1]
                for idx in top_k:
                    score = float(probs[idx])
                    if score > 0.001:
                        f.write(f"{pid}\t{top_terms[idx]}\t{score:.3f}\n")
                        
    print(f"✅ Submission saved to {Config.SUBMISSION}")

if __name__ == "__main__":
    run_inference()