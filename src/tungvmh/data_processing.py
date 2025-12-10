import os
import pandas as pd
import numpy as np
from datasets import Dataset
from Bio import SeqIO
import pickle
from tqdm import tqdm
from collections import defaultdict, Counter
import obonet
import networkx as nx

# ==================== CONFIGURATION ====================
CONFIG = {
    "top_n_terms": 10000,  # Lấy 10,000 terms phổ biến nhất theo yêu cầu
}
# =======================================================

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

TRAIN_SEQUENCES = os.path.join(TRAIN_DIR, "train_sequences.fasta")
TRAIN_TERMS = os.path.join(TRAIN_DIR, "train_terms.tsv")
TRAIN_TAXONOMY = os.path.join(TRAIN_DIR, "train_taxonomy.tsv")
TRAIN_OBO = os.path.join(TRAIN_DIR, "go-basic.obo")
TEST_SEQUENCES = os.path.join(TEST_DIR, "testsuperset.fasta")


def load_obo_graph_and_vocab():
    """
    Load OBO graph và lấy toàn bộ Vocab có trong file OBO.
    """
    print(f"Loading OBO ontology from {TRAIN_OBO}...")
    full_graph = obonet.read_obo(TRAIN_OBO)

    # Lấy toàn bộ danh sách GO terms có trong file OBO
    # Đây là Global Vocabulary (khoảng 40k terms)
    all_obo_terms = list(full_graph.nodes())
    print(f"Total GO terms in OBO file: {len(all_obo_terms)}")

    # Tạo graph rút gọn chỉ chứa quan hệ is_a để propagate (STRICT MODE)
    valid_relationships = {"is_a"}
    propagation_graph = nx.DiGraph()
    for u, v, key in full_graph.edges(keys=True):
        if key in valid_relationships:
            propagation_graph.add_edge(u, v)
    propagation_graph.add_nodes_from(full_graph.nodes())

    return propagation_graph, all_obo_terms


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


def propagate_and_select_top_k(protein_to_terms, graph, all_term_to_idx, top_k=10000):
    """
    1. Propagate terms trên tập train.
    2. Đếm tần suất.
    3. Chọn ra Top K terms phổ biến nhất để làm nhãn huấn luyện.
    """
    print("Propagating terms for training set...")

    # Cache tổ tiên để propagate nhanh hơn
    unique_raw_terms = set()
    for terms in protein_to_terms.values():
        unique_raw_terms.update(terms)

    ancestors_cache = {}
    for term in tqdm(unique_raw_terms, desc="Caching Ancestors"):
        if term in graph:
            # nx.descendants trả về tất cả node cha/ông
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

    # Chọn Top K (theo frequency cao nhất)
    # Most common trả về list [(term, count), ...]
    top_terms_list = term_counter.most_common(top_k)

    target_terms_set = set([t for t, c in top_terms_list])

    # Lưu danh sách Top K terms (để train_align.py dùng)
    top_10k_terms = [t for t, c in top_terms_list]

    print(f"Selected {len(top_10k_terms)} terms for training targets.")

    return protein_to_propagated, top_10k_terms, target_terms_set


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


def process_fasta(
    fasta_path,
    protein_to_propagated,
    pid_to_tax,
    all_term_to_idx,
    target_terms_set,
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

        # GO Terms (Chỉ xử lý cho Train)
        go_list = []
        go_ids = []

        if is_train:
            terms = protein_to_propagated.get(pid, [])
            if not terms:
                continue

            # Lọc: Chỉ giữ lại các terms nằm trong Top 10,000 target
            valid_terms = [t for t in terms if t in target_terms_set]

            if not valid_terms:
                continue

            go_list = valid_terms
            # Map sang GLOBAL ID (theo all_term_to_idx)
            # Lưu ý: train_align.py sẽ map từ Global ID -> Local ID (0-9999)
            go_ids = [all_term_to_idx[t] for t in valid_terms if t in all_term_to_idx]

        # Expand taxonomy
        for tax in set(taxs):
            data_buffer.append(
                {
                    "id": pid,
                    "seq": seq,
                    "taxonomy": tax,
                    "go_terms": go_list if is_train else None,
                    "go_terms_id": go_ids if is_train else None,
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


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load Graph & Full Vocab từ OBO
    graph, all_obo_terms = load_obo_graph_and_vocab()

    # Tạo mapping toàn cục cho tất cả terms
    all_term_to_idx = {t: i for i, t in enumerate(all_obo_terms)}

    # 2. Xử lý Train Labels: Propagate & Lấy Top K
    raw_protein_terms = get_raw_protein_terms()
    propagated_terms, top_10k_terms, target_terms_set = propagate_and_select_top_k(
        raw_protein_terms, graph, all_term_to_idx, top_k=CONFIG["top_n_terms"]
    )

    # 3. Save Vocab
    vocab_path = os.path.join(PROCESSED_DIR, "vocab.pkl")

    with open(vocab_path, "wb") as f:
        pickle.dump(
            {
                "top_10k_terms": top_10k_terms,  # List 10k terms phổ biến nhất
                "all_obo_terms": all_obo_terms,  # List 40k terms từ OBO
                "all_term_to_idx": all_term_to_idx,  # Mapping Full: Term -> ID (0-39999)
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
        all_term_to_idx,
        target_terms_set,
        train_out,
        is_train=True,
    )

    test_out = os.path.join(PROCESSED_DIR, "test.parquet")
    process_fasta(TEST_SEQUENCES, None, None, None, None, test_out, is_train=False)

    print("\nPROCESSING COMPLETE.")


if __name__ == "__main__":
    main()
