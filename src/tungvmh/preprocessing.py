import pandas as pd
import obonet
import networkx as nx
import pickle
import os
from tqdm import tqdm

# Paths
TRAIN_PARQUET_PATH = "data/processed2/train.parquet"
VOCAB_PKL_PATH = "data/processed/vocab.pkl"
OBO_PATH = "data/Train/go-basic.obo"
OUTPUT_PATH = "data/processed2/train_complete.parquet"


def load_data():
    print("Loading data...")
    if not os.path.exists(TRAIN_PARQUET_PATH):
        raise FileNotFoundError(f"{TRAIN_PARQUET_PATH} not found.")
    df = pd.read_parquet(TRAIN_PARQUET_PATH)

    if not os.path.exists(VOCAB_PKL_PATH):
        raise FileNotFoundError(f"{VOCAB_PKL_PATH} not found.")
    with open(VOCAB_PKL_PATH, "rb") as f:
        vocab_data = pickle.load(f)

    if isinstance(vocab_data, dict) and "term_to_idx" in vocab_data:
        vocab = vocab_data["term_to_idx"]
        print(f"Loaded vocab with {len(vocab)} terms.")
    else:
        vocab = vocab_data
        print(f"Loaded vocab with {len(vocab)} terms (direct mapping).")

    if not os.path.exists(OBO_PATH):
        raise FileNotFoundError(f"{OBO_PATH} not found.")
    print("Loading OBO ontology...")
    graph = obonet.read_obo(OBO_PATH)

    return df, vocab, graph


def get_ancestors(graph, term):
    """
    Get all ancestors of a GO term using the is_a relationship.
    """
    if term not in graph:
        return set()
    return nx.descendants(graph, term)


def propagate_and_filter(df, vocab, graph):
    print("Propagating GO terms...")

    vocab_set = set(vocab.keys())

    term_ancestors_cache = {}

    def process_terms(terms):
        if terms is None:
            return []

        all_terms = set(terms)
        for term in terms:
            if term in term_ancestors_cache:
                all_terms.update(term_ancestors_cache[term])
            elif term in graph:
                ancestors = nx.descendants(graph, term)
                term_ancestors_cache[term] = ancestors
                all_terms.update(ancestors)

    term_ancestors_cache = {}

    def process_terms(terms):
        if terms is None:
            return []

        all_terms = set(terms)
        for term in terms:
            if term in term_ancestors_cache:
                all_terms.update(term_ancestors_cache[term])
            elif term in graph:
                ancestors = nx.descendants(graph, term)
                term_ancestors_cache[term] = ancestors
                all_terms.update(ancestors)

        filtered_terms = [t for t in all_terms if t in vocab_set]
        return filtered_terms

    tqdm.pandas()
    df["go_terms"] = df["go_terms"].progress_apply(process_terms)

    return df


def main():
    try:
        df, vocab, graph = load_data()

        if "go_terms" not in df.columns:
            raise ValueError("Column 'go_terms' not found in train.parquet")

        df_processed = propagate_and_filter(df, vocab, graph)

        print(f"Saving to {OUTPUT_PATH}...")
        df_processed.to_parquet(OUTPUT_PATH)
        print("Done.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
