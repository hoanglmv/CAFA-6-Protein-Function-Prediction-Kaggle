import pickle
import os

VOCAB_PATH = "data/processed/vocab.pkl"


def check_vocab():
    if not os.path.exists(VOCAB_PATH):
        print(f"Error: {VOCAB_PATH} does not exist.")
        return

    print(f"Loading {VOCAB_PATH}...")
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    print("Keys in vocab:", vocab.keys())

    all_obo_terms = vocab.get("all_obo_terms", [])
    top_10k_terms = vocab.get("top_10k_terms", [])
    all_term_to_idx = vocab.get("all_term_to_idx", {})

    print(f"Total OBO terms: {len(all_obo_terms)}")
    print(f"Top 10k terms: {len(top_10k_terms)}")
    print(f"Global mapping size: {len(all_term_to_idx)}")

    # Check 1: all_obo_terms matches all_term_to_idx
    if len(all_obo_terms) != len(all_term_to_idx):
        print("WARNING: Mismatch between all_obo_terms and all_term_to_idx size.")
    else:
        print("Check 1 Passed: Global vocab size matches mapping.")

    # Check 2: Top 10k are in all terms
    missing = [t for t in top_10k_terms if t not in all_term_to_idx]
    if missing:
        print(f"WARNING: {len(missing)} terms from Top 10k are NOT in global vocab.")
    else:
        print("Check 2 Passed: All Top 10k terms are in Global Vocab.")

    # Check 3: Random sample check
    if top_10k_terms:
        sample = top_10k_terms[0]
        print(
            f"Sample Top 10k term: {sample}, Global ID: {all_term_to_idx.get(sample)}"
        )

    print("\nVerification Complete.")


if __name__ == "__main__":
    check_vocab()
