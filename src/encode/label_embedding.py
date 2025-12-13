import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import torch

# Global variable to store the model instance (Singleton pattern)
_model = None

def embed_labels(
    texts: List[str], 
    batch_size: int = 32, # Default nên > 1 để tận dụng GPU
    device: str = "cpu",  # <--- BẮT BUỘC CÓ để nhận tham số từ label_processing.py
    show_progress_bar: bool = True
) -> np.ndarray:
    """
    Embed a list of text descriptions using PubMedBERT via SentenceTransformer.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for processing.
        device: Device to run the model on ('cpu', 'cuda', etc.).
        show_progress_bar: Whether to show a progress bar.

    Returns:
        embeddings: Numpy array of embeddings with shape (len(texts), 768).
    """
    global _model
    
    # Tên model bạn muốn dùng
    model_name = "NeuML/pubmedbert-base-embeddings"

    if _model is None:
        print(f"📥 Loading SentenceTransformer model: {model_name}")
        print(f"⚙️  Device: {device}")
        # Khởi tạo model và đưa vào GPU
        _model = SentenceTransformer(model_name, device=device)

    # SentenceTransformer tự động xử lý tokenization và batching
    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        device=device,            # Đảm bảo chạy trên đúng device
        normalize_embeddings=True # Tốt cho các tác vụ similarity
    )

    return embeddings