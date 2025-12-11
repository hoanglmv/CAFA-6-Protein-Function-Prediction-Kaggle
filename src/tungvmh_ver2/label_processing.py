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

try:
    from src.encode.label_embedding import embed_labels
except ImportError:
    print("Error importing embed_labels. Make sure src.encode.label_embedding exists.")
    # Fallback or exit?
    # For now, let's assume it works or user will fix environment.
    pass

def main():
    # Paths
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed_ver2")
    PROCESSED2_DIR = os.path.join(DATA_DIR, "processed2_ver2")
    
    vocab_path = os.path.join(PROCESSED_DIR, "vocab.pkl")
    obo_path = os.path.join(DATA_DIR, "Train", "go-basic.obo")
    output_path = os.path.join(PROCESSED2_DIR, "label.parquet")

    os.makedirs(PROCESSED2_DIR, exist_ok=True)

    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    if not os.path.exists(vocab_path):
        print("Vocab file not found.")
        return

    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    # Get the 5000 terms
    top_terms = vocab_data.get("top_terms", [])
    term_to_idx = vocab_data.get("term_to_idx", {})
    
    if not top_terms:
        print("Error: 'top_terms' not found in vocab.pkl")
        return

    print(f"Total terms to embed: {len(top_terms)}")

    # Load GO ontology
    print(f"Loading GO ontology from {obo_path}...")
    graph = obonet.read_obo(obo_path)

    # Extract definitions
    print("Extracting rich definitions...")
    ids = []
    names = []
    definitions = []

    for term in tqdm(top_terms):
        # We use the index from 0-4999 as ID
        idx = term_to_idx[term]
        ids.append(idx)
        names.append(term) # GO ID

        # Create context: "NAME: DEFINITION"
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

    print(f"Embedding {len(definitions)} terms...")
    # Using batch_size=64 as in original script
    embeddings = embed_labels(definitions, batch_size=64, show_progress_bar=True)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "id": ids,       # Index 0-4999
            "name": names,   # GO ID
            "embedding": list(embeddings),
        }
    )

    # Save to parquet
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
