import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # Cần import thêm numpy


class ProteinGOAligner(nn.Module):
    def __init__(self, esm_dim=2560, go_emb_dim=768, joint_dim=512):
        super().__init__()

        # Nhánh 1: Project Protein
        self.prot_projector = nn.Sequential(
            nn.Linear(esm_dim, 1024),
            nn.BatchNorm1d(1024),  # Batch Norm rất quan trọng để ổn định training
            nn.ReLU(),
            nn.Linear(1024, joint_dim),
        )

        # Nhánh 2: Project GO Label
        self.go_projector = nn.Sequential(
            nn.Linear(go_emb_dim, joint_dim)
        )

        # log(1/0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, esm_embeddings, go_embeddings):
        prot_vec = self.prot_projector(esm_embeddings)
        go_vec = self.go_projector(go_embeddings)

        prot_vec = F.normalize(prot_vec, p=2, dim=1)
        go_vec = F.normalize(go_vec, p=2, dim=1)

        cosine_sim = torch.matmul(prot_vec, go_vec.T)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = cosine_sim * logit_scale

        return logits
