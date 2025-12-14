import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProteinGOAligner(nn.Module):
    def __init__(self, esm_dim=2564, go_emb_dim=768, joint_dim=512, num_classes=None):
        super().__init__()

        # esm_dim=2564 (2560 ESM + 4 Tax)
        self.prot_input_dim = esm_dim - 4  # 2560
        self.tax_dim = 4

        # ==================== 1. SOFT TAXONOMY ROUTER ====================
        self.tax_router = nn.Sequential(
            nn.Linear(self.tax_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.Sigmoid(),
        )

        # ==================== 2. PROTEIN ENCODER ====================
        self.prot_fc1 = nn.Linear(self.prot_input_dim, 1024)
        self.prot_bn1 = nn.BatchNorm1d(1024)
        self.prot_fc2 = nn.Linear(1024, joint_dim)

        # ==================== 3. GO TERM ENCODER ====================
        self.go_projector = nn.Sequential(
            nn.Linear(go_emb_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, joint_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(14))

        # ==================== 4. LEARNABLE BIAS ====================
        # Bias for each GO term to capture prior probability
        if num_classes is not None:
            self.go_bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.go_bias = None

    def forward(self, prot_features, go_embeddings):
        """
        prot_features: [Batch, 2564]
        """

        # 1. Tách Input
        esm_emb = prot_features[:, :-4]  # [Batch, 2560]
        tax_vec = prot_features[:, -4:]  # [Batch, 4]

        # BƯỚC 1: Tính Feature Protein Gốc
        prot_h = self.prot_fc1(esm_emb)  # [Batch, 1024]
        prot_h = self.prot_bn1(prot_h)
        prot_h = F.gelu(prot_h)

        # BƯỚC 2: Tính Soft Gate từ Taxonomy
        gate = self.tax_router(tax_vec)

        # BƯỚC 3: SOFT ROUTING (Feature Modulation)
        prot_routed = prot_h * gate

        # BƯỚC 4: Final Projection
        prot_vec = self.prot_fc2(prot_routed)

        # Xử lý nhánh GO Term
        go_vec = self.go_projector(go_embeddings)

        # Normalize & Cosine
        prot_vec = F.normalize(prot_vec, p=2, dim=1)
        go_vec = F.normalize(go_vec, p=2, dim=1)

        cosine_sim = torch.matmul(prot_vec, go_vec.T)
        logit_scale = self.logit_scale.exp().clamp(max=100)

        logits = cosine_sim * logit_scale

        # Add Bias if available
        if self.go_bias is not None:
            logits = logits + self.go_bias

        return logits
