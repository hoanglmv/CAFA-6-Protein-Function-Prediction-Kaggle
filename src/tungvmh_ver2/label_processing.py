import os
import pickle
import obonet
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

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
    # Import module Node2Vec vừa tạo
    from src.encode.graph_embedding import generate_node2vec_embeddings 
except ImportError:
    print("⚠️ Error importing generate_node2vec_embeddings. Make sure src/encode/graph_embedding.py exists.")

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
    # PHẦN 1: TEXT EMBEDDING (BioBERT/ESM)
    # ==========================================
    print("\n--- Processing Text Embeddings ---")
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
    text_embeddings = embed_labels(definitions, batch_size=64, show_progress_bar=True)
    
    # ==========================================
    # PHẦN 2: NODE EMBEDDING (Node2Vec)
    # ==========================================
    print("\n--- Processing Graph Node Embeddings ---")
    
    # Cấu hình Node2Vec (bạn có thể chỉnh dimensions tùy ý, thường là 64 hoặc 128)
    NODE_DIM = 64 
    
    # Gọi hàm từ module mới
    # Lưu ý: Hàm này chạy trên toàn bộ đồ thị GO để lấy ngữ cảnh đầy đủ, 
    # sau đó ta mới lọc ra các top_terms cần thiết.
    full_graph_embeddings = generate_node2vec_embeddings(
        graph, 
        dimensions=NODE_DIM, 
        walk_length=30, 
        num_walks=100, 
        workers=4
    )
    
    # Map kết quả từ full graph vào danh sách top_terms
    node_embeddings_list = []
    for term in top_terms:
        if term in full_graph_embeddings:
            node_embeddings_list.append(full_graph_embeddings[term])
        else:
            # Fallback nếu term không có trong graph (hiếm gặp)
            node_embeddings_list.append(np.zeros(NODE_DIM, dtype=np.float32))

    # ==========================================
    # SAVE RESULT
    # ==========================================
    # Create DataFrame
    df = pd.DataFrame(
        {
            "id": ids,            # Index 0-4999
            "name": names,        # GO ID (e.g., GO:0005515)
            "embedding": list(text_embeddings),       # Text vector (768 or 1280 dim)
            "node_embedding": node_embeddings_list    # Graph vector (64 dim)
        }
    )

    print(f"💾 Saving to {output_path}...")
    # Lưu parquet hỗ trợ lưu cột chứa list/array
    df.to_parquet(output_path)
    
    # Print sample to verify
    print("\nSample Data:")
    print(df.head(2))
    print(f"Shape: {df.shape}")
    print("✅ Done!")

if __name__ == "__main__":
    main()