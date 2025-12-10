import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import torch

# Global variable to store the model instance
_model = None




def embed_labels(
    texts: List[str], batch_size: int = 1, show_progress_bar: bool = True
) -> np.ndarray:
    """
    Embed a list of text descriptions using PubMedBERT.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for processing. Defaults to 1 as requested.
        show_progress_bar: Whether to show a progress bar.

    Returns:
        embeddings: Numpy array of embeddings with shape (len(texts), 768).
    """
    global _model
    if _model is None:
        model_name = "NeuML/pubmedbert-base-embeddings"
        _model = SentenceTransformer(model_name)

    # SentenceTransformer handles batching and device automatically
    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )

    return embeddings
