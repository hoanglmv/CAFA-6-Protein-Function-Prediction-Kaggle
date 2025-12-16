import os
import pandas as pd
import numpy as np
from Bio import SeqIO
import pickle
from tqdm import tqdm
from collections import defaultdict, Counter
import obonet
import networkx as nx

# ==================== CONFIGURATION ====================
CONFIG = {
    "top_n_terms": 5000, # Giảm xuống 5000 để mô hình tập trung hơn (CAFA thường chỉ đánh giá top terms)
}
# =======================================================

# Define paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_ver2")

TRAIN_SEQUENCES = os.path.join(TRAIN_DIR, "train_sequences.fasta")
TRAIN_TERMS = os.path.join(TRAIN_DIR, "train_terms.tsv")
TRAIN_TAXONOMY = os.path.join(TRAIN_DIR, "train_taxonomy.tsv")
TRAIN_OBO = os.path.join(TRAIN_DIR, "go-basic.obo")
TEST_SEQUENCES = os.path.join(TEST_DIR, "testsuperset.fasta")


def load_obo_graph():
    """
    Load OBO graph for propagation.
    """
    print(f"Loading OBO ontology from {TRAIN_OBO}...")
    full_graph = obonet.read_obo(TRAIN_OBO)

    # Create reduced graph with only is_a relationships for propagation
    valid_relationships = {"is_a"}
    propagation_graph = nx.DiGraph()
    for u, v, key in full_graph.edges(keys=True):
        if key in valid_relationships:
            propagation_graph.add_edge(u, v)
    propagation_graph.add_nodes_from(full_graph.nodes())

    return propagation_graph


def get_raw_protein_terms():
    print("Reading raw protein terms...")
    protein_to_terms = defaultdict(list)
    if not os.path.exists(TRAIN_TERMS):
        raise FileNotFoundError(f"Missing file: {TRAIN_TERMS}")

    chunk_size = 100000
    for chunk in pd.read_csv(TRAIN_TERMS, sep="\t", chunksize=chunk_size):
        for pid, group in chunk.groupby("EntryID"):
            protein_to_terms[pid].extend(group["term"].tolist())

    return protein_to_terms


def propagate_and_select_top_k(protein_to_terms, graph, top_k=5000):
    """
    1. Propagate terms on training set.
    2. Count frequency.
    3. Select Top K most common terms.
    4. Create mapping term -> index (0 to K-1).
    """
    print("Propagating terms for training set...")

    # Cache ancestors for faster propagation
    unique_raw_terms = set()
    for terms in protein_to_terms.values():
        unique_raw_terms.update(terms)

    ancestors_cache = {}
    for term in tqdm(unique_raw_terms, desc="Caching Ancestors"):
        if term in graph:
            ancestors = nx.descendants(graph, term)
            ancestors.add(term)
            ancestors_cache[term] = ancestors
        else:
            ancestors_cache[term] = {term}

    # Propagate & Count
    protein_to_propagated = {}
    term_counter = Counter()

    for pid, raw_terms in tqdm(protein_to_terms.items(), desc="Propagating Proteins"):
        propagated_set = set()
        for term in raw_terms:
            if term in ancestors_cache:
                propagated_set.update(ancestors_cache[term])
            else:
                propagated_set.add(term)

        protein_to_propagated[pid] = list(propagated_set)
        term_counter.update(propagated_set)

    print(f"Total unique propagated terms in Train set: {len(term_counter)}")

    # Select Top K
    top_terms_list = term_counter.most_common(top_k)
    top_terms = [t for t, c in top_terms_list]

    # Create mapping: Term -> Index (0 to K-1)
    term_to_idx = {t: i for i, t in enumerate(top_terms)}

    print(f"Selected {len(top_terms)} terms for training targets.")

    return protein_to_propagated, top_terms, term_to_idx


def get_protein_taxonomies():
    print("Loading taxonomies...")
    pid_to_tax = defaultdict(list)
    if os.path.exists(TRAIN_TAXONOMY):
        for chunk in pd.read_csv(
            TRAIN_TAXONOMY, sep="\t", header=None, names=["id", "tax"], chunksize=100000
        ):
            for _, row in chunk.iterrows():
                pid_to_tax[row["id"]].append(int(row["tax"]))
    return dict(pid_to_tax)


def save_parquet(data, path):
    df = pd.DataFrame(data)
    if not os.path.exists(path):
        df.to_parquet(path, index=False, engine="pyarrow")
    else:
        import pyarrow.parquet as pq
        import pyarrow as pa

        table = pa.Table.from_pandas(df)
        existing = pq.read_table(path)
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, path)


def process_fasta(
    fasta_path,
    protein_to_propagated,
    pid_to_tax,
    term_to_idx,
    output_path,
    is_train=True,
):
    print(f"Processing sequences from {fasta_path}...")

    data_buffer = []
    batch_size = 5000
    processed_count = 0

    if os.path.exists(output_path):
        os.remove(output_path)

    for record in tqdm(SeqIO.parse(fasta_path, "fasta")):
        # Parse ID
        head = record.id
        if "|" in head:
            parts = head.split("|")
            pid = parts[1] if len(parts) >= 2 else head
        else:
            pid = head

        seq = str(record.seq)

        # Taxonomy
        if is_train:
            taxs = pid_to_tax.get(pid, [0])
        else:
            parts = record.description.split()
            tax = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            taxs = [tax]

        # GO Terms (Only for Train)
        go_ids = []

        if is_train:
            terms = protein_to_propagated.get(pid, [])
            if not terms:
                continue

            # Filter and Map to Indices (0-4999)
            go_ids = [term_to_idx[t] for t in terms if t in term_to_idx]

            if not go_ids:
                continue

        # Expand taxonomy
        for tax in set(taxs):
            data_buffer.append(
                {
                    "id": pid,
                    "seq": seq,
                    "taxonomy": tax,
                    # "go_terms": terms if is_train else None, # Optional: Don't save raw terms to save space
                    "go_terms_id": (
                        go_ids if is_train else None
                    ),  # List of indices 0-4999
                }
            )

        if len(data_buffer) >= batch_size:
            save_parquet(data_buffer, output_path)
            processed_count += len(data_buffer)
            data_buffer = []

    if data_buffer:
        save_parquet(data_buffer, output_path)
        processed_count += len(data_buffer)

    print(f"Finished {output_path}: {processed_count} samples.")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load Graph
    graph = load_obo_graph()

    # 2. Process Train Labels: Propagate & Select Top K
    raw_protein_terms = get_raw_protein_terms()
    propagated_terms, top_terms, term_to_idx = propagate_and_select_top_k(
        raw_protein_terms, graph, top_k=CONFIG["top_n_terms"]
    )

    # --- NEW: Build Parent Map for Constraint Checking ---
    # Map: Index con -> List các Index cha
    print("Building Parent Map for constraints...")
    term_parents_indices = {}
    for child_term, child_idx in term_to_idx.items():
        if child_term in graph:
            # Lấy cha trực tiếp (hoặc tổ tiên nếu cần, ở đây lấy cha trực tiếp để tiết kiệm)
            parents = list(graph.successors(child_term)) 
            parent_indices = [term_to_idx[p] for p in parents if p in term_to_idx]
            if parent_indices:
                term_parents_indices[child_idx] = parent_indices
    # -----------------------------------------------------

    # 3. Save Vocab
    vocab_path = os.path.join(PROCESSED_DIR, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(
            {
                "top_terms": top_terms,  # List of 5000 terms
                "term_to_idx": term_to_idx,  # Mapping Term -> Index (0-4999)
                "term_parents_indices": term_parents_indices, # NEW: Để check True Path Rule
                "config": CONFIG,
            },
            f,
        )
    print(f"Vocab saved to {vocab_path}")

    # 4. Process Sequences -> Parquet
    pid_to_tax = get_protein_taxonomies()

    train_out = os.path.join(PROCESSED_DIR, "train.parquet")
    process_fasta(
        TRAIN_SEQUENCES,
        propagated_terms,
        pid_to_tax,
        term_to_idx,
        train_out,
        is_train=True,
    )

    test_out = os.path.join(PROCESSED_DIR, "test.parquet")
    process_fasta(TEST_SEQUENCES, None, None, None, test_out, is_train=False)

    print("\nPROCESSING COMPLETE.")


if __name__ == "__main__":
    main()