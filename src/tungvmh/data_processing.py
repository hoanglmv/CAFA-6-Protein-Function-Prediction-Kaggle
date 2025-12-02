import os
import pandas as pd
import numpy as np
from datasets import Dataset
from Bio import SeqIO

# Define paths
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

TRAIN_SEQUENCES = os.path.join(TRAIN_DIR, "train_sequences.fasta")
TRAIN_TERMS = os.path.join(TRAIN_DIR, "train_terms.tsv")
TRAIN_TAXONOMY = os.path.join(TRAIN_DIR, "train_taxonomy.tsv")
TEST_SEQUENCES = os.path.join(TEST_DIR, "testsuperset.fasta")
TEST_TAXONOMY = os.path.join(TEST_DIR, "testsuperset-taxon-list.tsv")


def parse_fasta_biopython(fasta_file, is_train=True):
    """
    Parses a FASTA file using Biopython and returns a list of dicts.
    """
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.description
        seq = str(record.seq)

        if is_train:
            # Train header example: sp|P9WHI7|RECN_MYCT ...
            # We need P9WHI7
            try:
                # Biopython parses record.id as the first word before space.
                # For Swiss-Prot, it's usually sp|ID|Name
                parts = record.id.split("|")
                if len(parts) >= 2:
                    pid = parts[1]
                else:
                    pid = record.id
            except:
                pid = record.id

            data.append({"id": pid, "seq": seq})
        else:
            # Test header example: A0A0C5B5G6 9606
            # record.id will be A0A0C5B5G6
            # record.description will be "A0A0C5B5G6 9606"
            pid = record.id

            # Extract taxonomy from description if available
            # Description: "ID TAXID"
            parts = record.description.split()
            tax = None
            if len(parts) > 1:
                try:
                    tax = int(parts[1])
                except:
                    pass

            data.append({"id": pid, "seq": seq, "taxonomy": tax})

    return data


def load_data():
    print("Loading training sequences with Biopython...")
    train_data = parse_fasta_biopython(TRAIN_SEQUENCES, is_train=True)
    df_train_seq = pd.DataFrame(train_data)

    print("Loading training taxonomy...")
    df_train_tax = pd.read_csv(
        TRAIN_TAXONOMY,
        sep="\t",
        header=None,
        names=["id", "taxonomy"],
        dtype={"taxonomy": int},
    )

    print("Loading training terms...")
    df_train_terms = pd.read_csv(TRAIN_TERMS, sep="\t")

    # Create GO term vocabulary
    print("Creating GO term vocabulary...")
    unique_terms = sorted(df_train_terms["term"].unique())
    term_to_idx = {term: i for i, term in enumerate(unique_terms)}
    print(f"Vocabulary size: {len(unique_terms)}")

    # Group terms by protein
    print("Grouping terms by protein...")
    protein_terms = df_train_terms.groupby("EntryID")["term"].apply(list).to_dict()

    # Process Training Data
    print("Processing training data...")
    # Merge sequences and taxonomy
    # Ensure IDs match.
    # df_train_seq has 'id' extracted from FASTA.
    # df_train_tax has 'id' (EntryID).

    # Merge seq and tax
    df_train = pd.merge(df_train_seq, df_train_tax, on="id", how="inner")

    # Add terms
    # Map ID to terms using the dictionary
    df_train["go_terms"] = df_train["id"].map(protein_terms)

    # Handle proteins with no terms (if any after merge)
    # The prompt says "This file contains only sequences for proteins with annotations in the dataset".
    # But if map returns NaN, we should handle it.
    df_train["go_terms"] = df_train["go_terms"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Create binary vectors
    print("Creating binary vectors for training data...")
    num_classes = len(unique_terms)

    def to_binary_vector(terms):
        vec = np.zeros(num_classes, dtype=np.int8)
        if not terms:
            return vec.tolist()
        indices = [term_to_idx[t] for t in terms if t in term_to_idx]
        vec[indices] = 1
        return vec.tolist()

    df_train["go_terms_id"] = df_train["go_terms"].apply(to_binary_vector)

    # Process Test Data
    print("Loading test sequences with Biopython...")
    test_data = parse_fasta_biopython(TEST_SEQUENCES, is_train=False)
    df_test = pd.DataFrame(test_data)

    # Fill missing columns for test
    df_test["go_terms"] = None
    df_test["go_terms_id"] = None

    # Ensure types
    df_train["taxonomy"] = df_train["taxonomy"].astype(int)
    # df_test["taxonomy"] is already int or None/NaN.
    # If we want to enforce int for parquet, we might need to handle NaNs if any exist.
    # But parquet handles nullable int.

    # Select columns
    cols = ["id", "seq", "taxonomy", "go_terms", "go_terms_id"]
    df_train = df_train[cols]
    df_test = df_test[cols]

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    # Create Datasets
    print("Creating Hugging Face Datasets...")
    ds_train = Dataset.from_pandas(df_train)
    ds_test = Dataset.from_pandas(df_test)

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_out = os.path.join(PROCESSED_DIR, "train.parquet")
    test_out = os.path.join(PROCESSED_DIR, "test.parquet")

    print(f"Saving to {train_out}...")
    ds_train.to_parquet(train_out)

    print(f"Saving to {test_out}...")
    ds_test.to_parquet(test_out)

    print("Done!")

    # Verify and Log
    print("Verifying and logging...")
    LOGS_DIR = "logs"
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Read back
    df_train_check = pd.read_parquet(train_out)
    df_test_check = pd.read_parquet(test_out)

    log_file = os.path.join(LOGS_DIR, "processed_data_preview.txt")
    with open(log_file, "w") as f:
        f.write("=== TRAIN DATA (First 15 rows) ===\n")
        f.write(df_train_check.head(15).to_string())
        f.write("\n\n")
        f.write("=== TEST DATA (First 15 rows) ===\n")
        f.write(df_test_check.head(15).to_string())
        f.write("\n")

    print(f"Logged preview to {log_file}")


if __name__ == "__main__":
    load_data()
