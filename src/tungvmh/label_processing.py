import os
import pickle
import obonet
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from encode.label_embedding import embed_labels
from utils.mapping import GOTermMapper


def main():
    # Paths
    vocab_path = "data/processed/vocab.pkl"
    obo_path = "data/Train/go-basic.obo"
    output_path = "data/processed2/label.parquet"

    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    # Lấy danh sách toàn bộ 40k terms
    all_obo_terms = vocab_data.get("all_obo_terms", [])
    if not all_obo_terms:
        print("Error: 'all_obo_terms' not found in vocab.pkl")
        return

    print(f"Total terms to embed: {len(all_obo_terms)}")

    # Load GO ontology
    print(f"Loading GO ontology from {obo_path}...")
    graph = obonet.read_obo(obo_path)

    # Extract definitions
    print("Extracting rich definitions...")
    ids = []
    names = []
    definitions = []

    for idx, term in enumerate(tqdm(all_obo_terms)):
        ids.append(idx)  # Global Index (0-39999)
        names.append(term)  # GO ID (GO:xxxxxxx)

        # [SOTA IMPROVEMENT] Tạo ngữ cảnh đầy đủ: "NAME: DEFINITION"
        text_content = term  # Mặc định

        if term in graph:
            node = graph.nodes[term]
            name_str = node.get("name", "")
            def_str = node.get("def", "")

            # Làm sạch chuỗi definition
            if def_str.startswith('"'):
                def_str = def_str.split('"')[1]

            # Kết hợp
            if name_str and def_str:
                text_content = f"{name_str}: {def_str}"
            elif name_str:
                text_content = name_str
            elif def_str:
                text_content = def_str

        # Fallback nếu không tìm thấy trong graph (hiếm)
        definitions.append(text_content)

    print(f"Embedding {len(definitions)} terms...")
    # Batch size lớn hơn chút để nhanh hơn nếu GPU cho phép
    embeddings = embed_labels(definitions, batch_size=64, show_progress_bar=True)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "id": ids,  # Index trong full vocab (0-39999)
            "name": names,  # GO ID (GO:xxxxxxx)
            "embedding": list(embeddings),
        }
    )

    # Save to parquet
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
