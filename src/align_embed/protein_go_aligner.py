import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProteinGOAligner(nn.Module):
    def __init__(self, esm_dim=2560, go_emb_dim=768):
        super().__init__()
        self.prot_projector = nn.Sequential(
            nn.Linear(esm_dim, 2048),  # Giữ chiều lớn để không mất tin
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, go_emb_dim),  # Output ra đúng 768 (bằng dim của GO)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, esm_embeddings, go_embeddings):
        """
        esm_embeddings: [Batch_Size, 2560]
        go_embeddings:  [Num_GO_Terms, 768] (hoặc Batch_GO, 768)
        """

        # 1. Project Protein về không gian 768
        prot_vec = self.prot_projector(esm_embeddings)  # -> [Batch, 768]

        # 2. GO Vector giữ nguyên (Chỉ chuẩn hóa)
        go_vec = go_embeddings  # -> [Num_Labels, 768]

        # 3. Normalize (Bắt buộc cho Cosine)
        prot_vec = F.normalize(prot_vec, p=2, dim=1)
        go_vec = F.normalize(go_vec, p=2, dim=1)

        # 4. Tính Similarity
        cosine_sim = torch.matmul(prot_vec, go_vec.T)

        # 5. Scale logits
        logit_scale = self.logit_scale.exp().clamp(max=50)
        logits = cosine_sim * logit_scale

        return logits
