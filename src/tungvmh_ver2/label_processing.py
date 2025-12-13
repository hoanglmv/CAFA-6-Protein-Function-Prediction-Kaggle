import os
import pickle
import obonet
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import torch  # <--- Thêm thư viện torch để check GPU

# Add src to path to import modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- IMPORT MODULES ---
try:
    from src.encode.label_embedding import embed_labels # Text Embedding
except ImportError:
    print("⚠️ Error importing embed_labels. Text embeddings might fail.")

try:
    # Import module Node2Vec
    from src.encode.graph_embedding import generate_node2vec_embeddings 
except ImportError:
    print("⚠️ Error importing generate_node2vec_embeddings. Make sure src/encode/graph_embedding.py exists.")

# --- CONFIG DEVICE ---
# Kiểm tra xem có GPU không
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

def main():
    # Paths
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed_ver2")
    PROCESSED2_DIR = os.path.join(DATA_DIR, "processed2_ver2")
    
    vocab_path = os.path.join(PROCESSED_DIR, "vocab.pkl")
    obo_path = os.path.join(DATA_DIR, "Train", "go-basic.obo")
    output_path = os.path.join(PROCESSED2_DIR, "label.parquet")

    os.makedirs(PROCESSED2_DIR, exist_ok=True)

    # 1. Load vocabulary
    print(f"📖 Loading vocabulary from {vocab_path}...")
    if not os.path.exists(vocab_path):
        print("❌ Vocab file not found.")
        return

    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    # Get the terms
    top_terms = vocab_data.get("top_terms", [])
    term_to_idx = vocab_data.get("term_to_idx", {})
    
    if not top_terms:
        print("❌ Error: 'top_terms' not found in vocab.pkl")
        return

    print(f"🔢 Total terms to embed: {len(top_terms)}")

    # 2. Load GO ontology (Graph)
    print(f"🕸️ Loading GO ontology from {obo_path}...")
    graph = obonet.read_obo(obo_path)

    # ==========================================
    # PHẦN 1: TEXT EMBEDDING (GPU ACCELERATED)
    # ==========================================
    print(f"\n--- Processing Text Embeddings on {DEVICE.upper()} ---")
    ids = []
    names = []
    definitions = []

    for term in tqdm(top_terms, desc="Extracting Definitions"):
        idx = term_to_idx[term]
        ids.append(idx)
        names.append(term) # GO ID

        # Extract text context
        text_content = term 
        if term in graph:
            node = graph.nodes[term]
            name_str = node.get("name", "")
            def_str = node.get("def", "")
            if def_str.startswith('"'):
                def_str = def_str.split('"')[1]

            if name_str and def_str:
                text_content = f"{name_str}: {def_str}"
            elif name_str:
                text_content = name_str
            elif def_str:
                text_content = def_str
        
        definitions.append(text_content)

    print(f"🔤 Embedding {len(definitions)} terms (Text)...")
    
    # --- UPDATE: Thêm tham số device và tăng batch_size ---
    # Với RTX 3060, batch_size=256 chạy rất mượt cho model MiniLM
    text_embeddings = embed_labels(
        definitions, 
        batch_size=256, 
        show_progress_bar=True,
        device=DEVICE    # Truyền device vào hàm
    )
    
    # ==========================================
    # PHẦN 2: NODE EMBEDDING (CPU - Gensim)
    # ==========================================
    print("\n--- Processing Graph Node Embeddings (Node2Vec usually runs on CPU) ---")
    
    # Cấu hình Node2Vec
    NODE_DIM = 64 
    
    # Node2Vec (Gensim) chạy trên CPU, ta tận dụng tối đa số core
    num_cpus = os.cpu_count()
    workers = max(1, num_cpus - 1) # Chừa lại 1 core cho hệ thống
    print(f"Using {workers} CPU workers for Random Walks...")

    full_graph_embeddings = generate_node2vec_embeddings(
        graph, 
        dimensions=NODE_DIM, 
        walk_length=30, 
        num_walks=100, 
        workers=workers # Tối ưu số luồng CPU
    )
    
    # Map kết quả từ full graph vào danh sách top_terms
    node_embeddings_list = []
    for term in top_terms:
        if term in full_graph_embeddings:
            node_embeddings_list.append(full_graph_embeddings[term])
        else:
            node_embeddings_list.append(np.zeros(NODE_DIM, dtype=np.float32))

    # ==========================================
    # SAVE RESULT
    # ==========================================
    # Create DataFrame
    df = pd.DataFrame(
        {
            "id": ids,            # Index 0-4999
            "name": names,        # GO ID
            "embedding": list(text_embeddings),       # Text vector
            "node_embedding": node_embeddings_list    # Graph vector
        }
    )

    print(f"💾 Saving to {output_path}...")
    df.to_parquet(output_path)
    
    # Print sample to verify
    print("\nSample Data:")
    print(df.head(2))
    print(f"Shape: {df.shape}")
    print("✅ Done!")

if __name__ == "__main__":
    main()