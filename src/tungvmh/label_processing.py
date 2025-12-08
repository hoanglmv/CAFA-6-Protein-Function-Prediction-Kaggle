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
    mapper = GOTermMapper(vocab_path)
    vocab_terms = mapper.get_all_terms()
    print(f"Loaded {len(vocab_terms)} terms from vocabulary.")

    # Load GO ontology
    print(f"Loading GO ontology from {obo_path}...")
    graph = obonet.read_obo(obo_path)
    print(f"Loaded ontology with {len(graph)} terms.")

    # Extract definitions
    print("Extracting definitions...")
    ids = []
    names = []
    definitions = []

    for idx, term in enumerate(tqdm(vocab_terms)):
        ids.append(idx)
        names.append(term)

        if term in graph:
            node = graph.nodes[term]
            if "def" in node:
                print("Definition found for term:", term)
                def_str = node["def"]
                if def_str.startswith('"'):
                    def_text = def_str.split('"')[1]
                else:
                    def_text = def_str
                definitions.append(def_text)
            elif "name" in node:
                print("Name found for term:", term)
                definitions.append(node["name"])
            else:
                print("No definition or name found for term:", term)
                definitions.append(term)  # Fallback to ID if nothing else
        else:
            print(
                f"Warning: Term {term} not found in ontology. Using term ID as definition."
            )
            definitions.append(term)

    # Embed definitions
    print(f"Embedding {len(definitions)} definitions...")
    # Use batch_size=1 and show_progress_bar=True as requested/planned
    embeddings = embed_labels(definitions, batch_size=1, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Create DataFrame
    print("Creating DataFrame...")
    # Convert embeddings to list of arrays for DataFrame storage if needed,
    # but parquet handles numpy arrays well if we are careful.
    # The user requested specific columns: id, name, embedding (np array 768)

    # To store numpy arrays in a parquet column, we can keep them as objects or lists.
    # Let's verify what works best. Usually list of lists or just numpy array column.
    # However, a column of numpy arrays is often stored as a list of floats in parquet.
    # Let's try storing as list of floats to be safe and compatible.
    # Wait, user said "embedding (np array 768)".
    # If I save as parquet, it will likely be read back as numpy array if I use pyarrow.

    df = pd.DataFrame(
        {
            "id": ids,
            "name": names,
            "embedding": list(embeddings),  # Store as list of arrays
        }
    )

    # Save to parquet
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
