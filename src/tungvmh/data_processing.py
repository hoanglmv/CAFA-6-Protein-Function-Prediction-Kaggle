import os
import pandas as pd
import numpy as np
from datasets import Dataset
from Bio import SeqIO
import pickle
from tqdm import tqdm
from collections import defaultdict, Counter

# ==================== CONFIGURATION ====================
CONFIG = {
    # Set to None to use all terms, or set to a number (e.g., 1000, 5000)
    # to use only the top N most frequent terms
    "top_n_terms": 20000,  # None = all terms, or int like 1000, 5000, 10000
    # Minimum term frequency (optional filter)
    "min_term_frequency": 1,  # Only keep terms that appear at least this many times
}
# =======================================================

# Define paths
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

TRAIN_SEQUENCES = os.path.join(TRAIN_DIR, "train_sequences.fasta")
TRAIN_TERMS = os.path.join(TRAIN_DIR, "train_terms.tsv")
TRAIN_TAXONOMY = os.path.join(TRAIN_DIR, "train_taxonomy.tsv")
TEST_SEQUENCES = os.path.join(TEST_DIR, "testsuperset.fasta")


def analyze_term_frequency():
    """
    Analyze GO term frequency from train_terms.tsv
    Returns: Counter object with term frequencies
    """
    print("Analyzing GO term frequencies...")

    term_counter = Counter()
    chunk_size = 100000

    for chunk in pd.read_csv(TRAIN_TERMS, sep="\t", chunksize=chunk_size):
        # Count terms from the 'term' column
        term_counter.update(chunk["term"].tolist())

    print(f"Total unique terms found: {len(term_counter)}")
    print(f"Total term annotations: {sum(term_counter.values())}")

    return term_counter


def create_vocab_and_mappings(top_n=None, min_frequency=1):
    """
    Create GO term vocabulary and protein mappings efficiently.

    Args:
        top_n: If specified, only use top N most frequent terms
        min_frequency: Minimum frequency for a term to be included

    Returns: term_to_idx, protein_to_terms, protein_to_taxonomies, term_stats
    """
    print("Creating GO term vocabulary and mappings...")

    # Step 1: Analyze term frequencies
    term_counter = analyze_term_frequency()

    # Step 2: Select terms based on configuration
    if top_n is not None:
        print(f"\nSelecting top {top_n} most frequent terms...")
        selected_terms = [term for term, count in term_counter.most_common(top_n)]
        print(f"Selected {len(selected_terms)} terms")
    else:
        print(f"\nFiltering terms with minimum frequency: {min_frequency}")
        selected_terms = [
            term for term, count in term_counter.items() if count >= min_frequency
        ]
        print(f"Selected {len(selected_terms)} terms after filtering")

    # Convert to set for fast lookup
    selected_terms_set = set(selected_terms)

    # Step 3: Create vocabulary (sorted for consistency)
    selected_terms_sorted = sorted(selected_terms)
    term_to_idx = {term: i for i, term in enumerate(selected_terms_sorted)}

    print(f"\nFinal vocabulary size: {len(term_to_idx)}")

    # Step 4: Build protein->terms mapping (only include selected terms)
    print("Building protein-to-terms mapping...")
    protein_to_terms = defaultdict(list)
    chunk_size = 100000

    for chunk in pd.read_csv(TRAIN_TERMS, sep="\t", chunksize=chunk_size):
        # Filter to only selected terms
        chunk_filtered = chunk[chunk["term"].isin(selected_terms_set)]

        for protein_id, group in chunk_filtered.groupby("EntryID"):
            protein_to_terms[protein_id].extend(group["term"].tolist())

    # Remove duplicates in terms
    for pid in protein_to_terms:
        protein_to_terms[pid] = list(set(protein_to_terms[pid]))

    print(f"Proteins with at least one selected term: {len(protein_to_terms)}")

    # Step 5: Build protein->taxonomies mapping
    print("Building protein-to-taxonomy mapping...")
    protein_to_taxonomies = defaultdict(list)

    for chunk in pd.read_csv(
        TRAIN_TAXONOMY,
        sep="\t",
        header=None,
        names=["id", "taxonomy"],
        chunksize=chunk_size,
    ):
        for _, row in chunk.iterrows():
            protein_to_taxonomies[row["id"]].append(int(row["taxonomy"]))

    # Remove duplicates in taxonomies
    for pid in protein_to_taxonomies:
        protein_to_taxonomies[pid] = list(set(protein_to_taxonomies[pid]))

    # Step 6: Collect statistics
    term_stats = {
        "total_unique_terms": len(term_counter),
        "selected_terms": len(term_to_idx),
        "top_n_config": top_n,
        "min_frequency_config": min_frequency,
        "term_frequencies": {
            term: term_counter[term] for term in selected_terms_sorted
        },
    }

    return term_to_idx, dict(protein_to_terms), dict(protein_to_taxonomies), term_stats


def log_term_statistics(term_stats, log_dir="logs"):
    """
    Log detailed term statistics to a file
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "term_statistics.txt")

    with open(log_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GO TERM STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total unique terms in dataset: {term_stats['total_unique_terms']}\n")
        f.write(f"Selected terms for vocabulary: {term_stats['selected_terms']}\n")
        f.write(f"Top N configuration: {term_stats['top_n_config']}\n")
        f.write(
            f"Min frequency configuration: {term_stats['min_frequency_config']}\n\n"
        )

        f.write("=" * 80 + "\n")
        f.write("TOP 50 MOST FREQUENT TERMS\n")
        f.write("=" * 80 + "\n")

        term_freq = term_stats["term_frequencies"]
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)

        for i, (term, freq) in enumerate(sorted_terms[:50], 1):
            f.write(f"{i:3d}. {term:15s} - {freq:6d} occurrences\n")

        if len(sorted_terms) > 50:
            f.write(f"\n... and {len(sorted_terms) - 50} more terms\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("FREQUENCY DISTRIBUTION\n")
        f.write("=" * 80 + "\n")

        frequencies = list(term_freq.values())
        f.write(f"Mean frequency: {np.mean(frequencies):.2f}\n")
        f.write(f"Median frequency: {np.median(frequencies):.2f}\n")
        f.write(f"Min frequency: {np.min(frequencies)}\n")
        f.write(f"Max frequency: {np.max(frequencies)}\n")
        f.write(f"Std deviation: {np.std(frequencies):.2f}\n")

    print(f"Term statistics logged to {log_file}")


def process_train_sequences(
    fasta_file, term_to_idx, protein_to_terms, protein_to_taxonomies, output_file
):
    """
    Process training sequences.
    If a protein has multiple taxonomies, create multiple samples.
    """
    print(f"Processing {fasta_file}...")

    batch_size = 1000
    batch_data = []
    processed = 0
    skipped = 0
    skipped_no_taxonomy = 0
    skipped_no_terms = 0

    for record in tqdm(SeqIO.parse(fasta_file, "fasta")):
        # Extract protein ID
        parts = record.id.split("|")
        pid = parts[1] if len(parts) >= 2 else record.id

        # Get sequence
        seq = str(record.seq)

        # Get taxonomies (can be multiple)
        taxonomies = protein_to_taxonomies.get(pid, [])
        if not taxonomies:
            skipped += 1
            skipped_no_taxonomy += 1
            continue

        # Get GO terms
        terms = protein_to_terms.get(pid, [])
        if not terms:
            skipped += 1
            skipped_no_terms += 1
            continue

        # Convert terms to indices (only terms in vocabulary)
        term_indices = [term_to_idx[t] for t in terms if t in term_to_idx]

        # Filter terms to only those in vocabulary
        valid_terms = [t for t in terms if t in term_to_idx]

        if not valid_terms:
            skipped += 1
            skipped_no_terms += 1
            continue

        # Create one sample for each taxonomy
        for taxonomy in taxonomies:
            batch_data.append(
                {
                    "id": pid,
                    "seq": seq,
                    "taxonomy": taxonomy,
                    "go_terms": valid_terms,  # List[str] - only terms in vocab
                    "go_terms_id": term_indices,  # List[int]
                }
            )
            processed += 1

        # Write batch to disk
        if len(batch_data) >= batch_size:
            write_batch(batch_data, output_file, processed == len(batch_data))
            batch_data = []

    # Write remaining data
    if batch_data:
        write_batch(batch_data, output_file, False)

    print(f"Processed {processed} samples from sequences")
    print(f"Skipped {skipped} sequences total:")
    print(f"  - No taxonomy: {skipped_no_taxonomy}")
    print(f"  - No terms (or no terms in vocab): {skipped_no_terms}")
    return processed


def process_test_sequences(fasta_file, output_file):
    """
    Process test sequences.
    Extract taxonomy from FASTA header.
    """
    print(f"Processing {fasta_file}...")

    batch_size = 1000
    batch_data = []
    processed = 0

    for record in tqdm(SeqIO.parse(fasta_file, "fasta")):
        # Extract protein ID
        pid = record.id

        # Get sequence
        seq = str(record.seq)

        # Extract taxonomy from description
        # Description format: "A0A0C5B5G6 9606"
        parts = record.description.split()
        taxonomy = None
        if len(parts) > 1:
            try:
                taxonomy = int(parts[1])
            except:
                pass

        batch_data.append(
            {
                "id": pid,
                "seq": seq,
                "taxonomy": taxonomy,
                "go_terms": None,  # No labels for test
                "go_terms_id": None,  # No labels for test
            }
        )
        processed += 1

        # Write batch to disk
        if len(batch_data) >= batch_size:
            write_batch(batch_data, output_file, processed == len(batch_data))
            batch_data = []

    # Write remaining data
    if batch_data:
        write_batch(batch_data, output_file, False)

    print(f"Processed {processed} test sequences")
    return processed


def write_batch(batch_data, output_file, is_first_batch):
    """
    Write a batch of data to parquet file.
    Append mode for subsequent batches.
    """
    df = pd.DataFrame(batch_data)

    if is_first_batch or not os.path.exists(output_file):
        df.to_parquet(output_file, index=False, engine="pyarrow")
    else:
        # Append to existing file
        import pyarrow.parquet as pq
        import pyarrow as pa

        # Read existing
        table_existing = pq.read_table(output_file)

        # Convert new data to table
        table_new = pa.Table.from_pandas(df)

        # Concatenate
        table_combined = pa.concat_tables([table_existing, table_new])

        # Write back
        pq.write_table(table_combined, output_file)


def load_data():
    """
    Main function - process all data and create datasets.

    Output format:
    - id: str (protein ID)
    - seq: str (amino acid sequence)
    - taxonomy: int (taxon ID)
    - go_terms: List[str] (GO term IDs like "GO:0008150")
    - go_terms_id: List[int] (indices in vocabulary)
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Top N terms: {CONFIG['top_n_terms']}")
    print(f"Min term frequency: {CONFIG['min_term_frequency']}")
    print("=" * 80 + "\n")

    # Step 1: Create mappings with term selection
    term_to_idx, protein_to_terms, protein_to_taxonomies, term_stats = (
        create_vocab_and_mappings(
            top_n=CONFIG["top_n_terms"], min_frequency=CONFIG["min_term_frequency"]
        )
    )

    # Log term statistics
    log_term_statistics(term_stats)

    # Save vocab for later use
    vocab_file = os.path.join(PROCESSED_DIR, "vocab.pkl")
    with open(vocab_file, "wb") as f:
        pickle.dump(
            {
                "term_to_idx": term_to_idx,
                "idx_to_term": {v: k for k, v in term_to_idx.items()},
                "num_classes": len(term_to_idx),
                "config": CONFIG,
                "term_stats": term_stats,
            },
            f,
        )
    print(f"Saved vocabulary to {vocab_file}")

    # Step 2: Process train sequences
    train_out = os.path.join(PROCESSED_DIR, "train.parquet")
    process_train_sequences(
        TRAIN_SEQUENCES, term_to_idx, protein_to_terms, protein_to_taxonomies, train_out
    )

    # Step 3: Process test sequences
    test_out = os.path.join(PROCESSED_DIR, "test.parquet")
    process_test_sequences(TEST_SEQUENCES, test_out)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Train data: {train_out}")
    print(f"Test data: {test_out}")
    print(f"Vocabulary: {vocab_file}")

    # Convert to HuggingFace Datasets
    print("\nConverting to HuggingFace Datasets format...")
    df_train = pd.read_parquet(train_out)
    df_test = pd.read_parquet(test_out)

    ds_train = Dataset.from_pandas(df_train, preserve_index=False)
    ds_test = Dataset.from_pandas(df_test, preserve_index=False)

    # Save as HuggingFace datasets
    train_hf_out = os.path.join(PROCESSED_DIR, "train_dataset")
    test_hf_out = os.path.join(PROCESSED_DIR, "test_dataset")

    ds_train.save_to_disk(train_hf_out)
    ds_test.save_to_disk(test_hf_out)

    print(f"\nHuggingFace datasets saved:")
    print(f"  Train: {train_hf_out}")
    print(f"  Test: {test_hf_out}")

    # Log preview
    print("\nCreating preview log...")
    LOGS_DIR = "logs"
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_file = os.path.join(LOGS_DIR, "processed_data_preview.txt")
    with open(log_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET OVERVIEW\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  Top N terms: {CONFIG['top_n_terms']}\n")
        f.write(f"  Min term frequency: {CONFIG['min_term_frequency']}\n\n")

        f.write(f"Train samples: {len(df_train)}\n")
        f.write(f"Test samples: {len(df_test)}\n")
        f.write(f"Vocabulary size: {len(term_to_idx)}\n")
        f.write(f"Total terms in dataset: {term_stats['total_unique_terms']}\n\n")

        f.write("=" * 80 + "\n")
        f.write("TRAIN DATA (First 10 rows)\n")
        f.write("=" * 80 + "\n")
        for i in range(min(10, len(df_train))):
            row = df_train.iloc[i]
            f.write(f"\nSample {i+1}:\n")
            f.write(f"  ID: {row['id']}\n")
            f.write(f"  Sequence (first 50 chars): {row['seq'][:50]}...\n")
            f.write(f"  Taxonomy: {row['taxonomy']}\n")

            # Handle list/array for go_terms
            go_terms = row["go_terms"]
            if go_terms is not None and len(go_terms) > 0:
                f.write(f"  GO terms count: {len(go_terms)}\n")
                f.write(f"  GO terms (first 5): {list(go_terms)[:5]}\n")
            else:
                f.write(f"  GO terms count: 0\n")
                f.write(f"  GO terms (first 5): None\n")

            # Handle list/array for go_terms_id
            go_terms_id = row["go_terms_id"]
            if go_terms_id is not None and len(go_terms_id) > 0:
                f.write(f"  GO term indices (first 5): {list(go_terms_id)[:5]}\n")
            else:
                f.write(f"  GO term indices (first 5): None\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("TEST DATA (First 10 rows)\n")
        f.write("=" * 80 + "\n")
        for i in range(min(10, len(df_test))):
            row = df_test.iloc[i]
            f.write(f"\nSample {i+1}:\n")
            f.write(f"  ID: {row['id']}\n")
            f.write(f"  Sequence (first 50 chars): {row['seq'][:50]}...\n")
            f.write(f"  Taxonomy: {row['taxonomy']}\n")
            f.write(f"  GO terms: None\n")
            f.write(f"  GO term indices: None\n")

        # Statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICS\n")
        f.write("=" * 80 + "\n")

        if len(df_train) > 0:
            # Safe way to calculate average
            go_terms_lengths = []
            for terms in df_train["go_terms"]:
                if terms is not None and len(terms) > 0:
                    go_terms_lengths.append(len(terms))
                else:
                    go_terms_lengths.append(0)

            avg_terms = (
                sum(go_terms_lengths) / len(go_terms_lengths) if go_terms_lengths else 0
            )
            avg_seq_len = df_train["seq"].apply(len).mean()

            f.write(f"\nTrain set:\n")
            f.write(f"  Average GO terms per protein: {avg_terms:.2f}\n")
            f.write(f"  Average sequence length: {avg_seq_len:.2f}\n")
            f.write(f"  Unique proteins: {df_train['id'].nunique()}\n")
            f.write(f"  Unique taxonomies: {df_train['taxonomy'].nunique()}\n")

        if len(df_test) > 0:
            avg_seq_len_test = df_test["seq"].apply(len).mean()
            f.write(f"\nTest set:\n")
            f.write(f"  Average sequence length: {avg_seq_len_test:.2f}\n")
            f.write(f"  Unique proteins: {df_test['id'].nunique()}\n")
            f.write(f"  Unique taxonomies: {df_test['taxonomy'].nunique()}\n")

    print(f"Logged preview to {log_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Train samples: {len(df_train):,}")
    print(f"✓ Test samples: {len(df_test):,}")
    print(
        f"✓ GO terms vocabulary: {len(term_to_idx):,} (out of {term_stats['total_unique_terms']:,} total)"
    )
    print(f"✓ Top N terms config: {CONFIG['top_n_terms']}")
    print(f"✓ Min frequency config: {CONFIG['min_term_frequency']}")
    print(f"✓ Files saved in: {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    load_data()
