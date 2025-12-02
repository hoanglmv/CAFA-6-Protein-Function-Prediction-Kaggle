"""
utils/mapping.py

Utilities for mapping between GO term IDs and indices.
Load the vocabulary created during data processing and provide conversion functions.
"""

import pickle
import os
from typing import List, Union, Optional, Dict
import numpy as np
import torch


class GOTermMapper:
    """
    Mapper class for converting between GO term strings and indices.

    Usage:
        mapper = GOTermMapper("data/processed/vocab.pkl")

        # Convert terms to indices
        indices = mapper.terms_to_indices(["GO:0008150", "GO:0005575"])

        # Convert indices to terms
        terms = mapper.indices_to_terms([0, 15, 234])

        # Create binary vector
        binary = mapper.terms_to_binary(["GO:0008150", "GO:0005575"])
    """

    def __init__(self, vocab_path: str = "data/processed/vocab.pkl"):
        """
        Initialize the mapper by loading vocabulary.

        Args:
            vocab_path: Path to the vocabulary pickle file
        """
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"Vocabulary file not found at {vocab_path}. "
                "Please run data processing first."
            )

        with open(vocab_path, "rb") as f:
            vocab_data = pickle.load(f)

        self.term_to_idx: Dict[str, int] = vocab_data["term_to_idx"]
        self.idx_to_term: Dict[int, str] = vocab_data["idx_to_term"]
        self.num_classes: int = vocab_data["num_classes"]

        print(f"Loaded vocabulary with {self.num_classes} GO terms")

    def term_to_index(self, term: str) -> Optional[int]:
        """
        Convert a single GO term to its index.

        Args:
            term: GO term string (e.g., "GO:0008150")

        Returns:
            Index of the term, or None if not found
        """
        return self.term_to_idx.get(term)

    def index_to_term(self, idx: int) -> Optional[str]:
        """
        Convert a single index to its GO term.

        Args:
            idx: Index of the term

        Returns:
            GO term string, or None if not found
        """
        return self.idx_to_term.get(idx)

    def terms_to_indices(
        self, terms: List[str], skip_unknown: bool = True
    ) -> List[int]:
        """
        Convert a list of GO terms to indices.

        Args:
            terms: List of GO term strings
            skip_unknown: If True, skip terms not in vocabulary

        Returns:
            List of indices
        """
        if skip_unknown:
            return [self.term_to_idx[t] for t in terms if t in self.term_to_idx]
        else:
            indices = []
            for t in terms:
                idx = self.term_to_idx.get(t)
                if idx is None:
                    raise ValueError(f"Term {t} not found in vocabulary")
                indices.append(idx)
            return indices

    def indices_to_terms(
        self, indices: List[int], skip_unknown: bool = True
    ) -> List[str]:
        """
        Convert a list of indices to GO terms.

        Args:
            indices: List of term indices
            skip_unknown: If True, skip indices not in vocabulary

        Returns:
            List of GO term strings
        """
        if skip_unknown:
            return [self.idx_to_term[i] for i in indices if i in self.idx_to_term]
        else:
            terms = []
            for i in indices:
                term = self.idx_to_term.get(i)
                if term is None:
                    raise ValueError(f"Index {i} not found in vocabulary")
                terms.append(term)
            return terms

    def terms_to_binary(
        self, terms: List[str], return_numpy: bool = False, return_torch: bool = False
    ) -> Union[List[int], np.ndarray, torch.Tensor]:
        """
        Convert GO terms to binary vector.

        Args:
            terms: List of GO term strings
            return_numpy: If True, return as numpy array
            return_torch: If True, return as torch tensor

        Returns:
            Binary vector of shape (num_classes,)
        """
        indices = self.terms_to_indices(terms, skip_unknown=True)

        if return_torch:
            vec = torch.zeros(self.num_classes, dtype=torch.float32)
            if indices:
                vec[indices] = 1
            return vec
        elif return_numpy:
            vec = np.zeros(self.num_classes, dtype=np.float32)
            if indices:
                vec[indices] = 1
            return vec
        else:
            vec = [0] * self.num_classes
            for idx in indices:
                vec[idx] = 1
            return vec

    def indices_to_binary(
        self, indices: List[int], return_numpy: bool = False, return_torch: bool = False
    ) -> Union[List[int], np.ndarray, torch.Tensor]:
        """
        Convert indices to binary vector.

        Args:
            indices: List of term indices
            return_numpy: If True, return as numpy array
            return_torch: If True, return as torch tensor

        Returns:
            Binary vector of shape (num_classes,)
        """
        if return_torch:
            vec = torch.zeros(self.num_classes, dtype=torch.float32)
            if indices:
                vec[indices] = 1
            return vec
        elif return_numpy:
            vec = np.zeros(self.num_classes, dtype=np.float32)
            if indices:
                vec[indices] = 1
            return vec
        else:
            vec = [0] * self.num_classes
            for idx in indices:
                vec[idx] = 1
            return vec

    def binary_to_indices(
        self, binary_vec: Union[List, np.ndarray, torch.Tensor], threshold: float = 0.5
    ) -> List[int]:
        """
        Convert binary vector to list of indices.

        Args:
            binary_vec: Binary vector (can be list, numpy, or torch)
            threshold: Threshold for considering a value as positive

        Returns:
            List of indices where value >= threshold
        """
        if isinstance(binary_vec, torch.Tensor):
            binary_vec = binary_vec.cpu().numpy()
        elif isinstance(binary_vec, list):
            binary_vec = np.array(binary_vec)

        return np.where(binary_vec >= threshold)[0].tolist()

    def binary_to_terms(
        self, binary_vec: Union[List, np.ndarray, torch.Tensor], threshold: float = 0.5
    ) -> List[str]:
        """
        Convert binary vector to list of GO terms.

        Args:
            binary_vec: Binary vector (can be list, numpy, or torch)
            threshold: Threshold for considering a value as positive

        Returns:
            List of GO term strings
        """
        indices = self.binary_to_indices(binary_vec, threshold)
        return self.indices_to_terms(indices)

    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return self.num_classes

    def get_all_terms(self) -> List[str]:
        """Get all GO terms in the vocabulary (sorted by index)."""
        return [self.idx_to_term[i] for i in range(self.num_classes)]

    def contains_term(self, term: str) -> bool:
        """Check if a term exists in the vocabulary."""
        return term in self.term_to_idx

    def contains_index(self, idx: int) -> bool:
        """Check if an index exists in the vocabulary."""
        return idx in self.idx_to_term


def load_mapper(vocab_path: str = "data/processed/vocab.pkl") -> GOTermMapper:
    """
    Convenience function to load a GOTermMapper.

    Args:
        vocab_path: Path to vocabulary file

    Returns:
        GOTermMapper instance
    """
    return GOTermMapper(vocab_path)


# Example usage and testing
if __name__ == "__main__":
    # Initialize mapper
    mapper = GOTermMapper("data/processed/vocab.pkl")

    print("=" * 60)
    print("GO TERM MAPPER - EXAMPLES")
    print("=" * 60)

    # Example 1: Term to index
    print("\n1. Term to Index:")
    term = "GO:0008150"
    idx = mapper.term_to_index(term)
    print(f"   {term} -> {idx}")

    # Example 2: Index to term
    print("\n2. Index to Term:")
    print(f"   {idx} -> {mapper.index_to_term(idx)}")

    # Example 3: Multiple terms to indices
    print("\n3. Multiple Terms to Indices:")
    terms = ["GO:0008150", "GO:0005575", "GO:0003674"]
    indices = mapper.terms_to_indices(terms)
    print(f"   {terms}")
    print(f"   -> {indices}")

    # Example 4: Indices to terms
    print("\n4. Indices to Terms:")
    print(f"   {indices}")
    print(f"   -> {mapper.indices_to_terms(indices)}")

    # Example 5: Terms to binary vector (numpy)
    print("\n5. Terms to Binary Vector:")
    binary = mapper.terms_to_binary(terms, return_numpy=True)
    print(f"   Shape: {binary.shape}")
    print(f"   Non-zero positions: {np.where(binary > 0)[0].tolist()}")
    print(f"   Sum: {binary.sum()}")

    # Example 6: Binary to terms
    print("\n6. Binary Vector to Terms:")
    recovered_terms = mapper.binary_to_terms(binary)
    print(f"   {recovered_terms}")

    # Example 7: Check vocabulary info
    print("\n7. Vocabulary Info:")
    print(f"   Total GO terms: {mapper.get_vocab_size()}")
    print(f"   First 5 terms: {mapper.get_all_terms()[:5]}")

    # Example 8: Using with torch
    print("\n8. PyTorch Integration:")
    try:
        import torch

        binary_torch = mapper.terms_to_binary(terms, return_torch=True)
        print(f"   Torch tensor shape: {binary_torch.shape}")
        print(f"   Torch tensor dtype: {binary_torch.dtype}")
        print(f"   Non-zero count: {(binary_torch > 0).sum().item()}")
    except ImportError:
        print("   PyTorch not installed, skipping...")

    print("\n" + "=" * 60)
