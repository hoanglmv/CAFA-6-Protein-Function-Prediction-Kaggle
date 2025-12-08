import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinGOAligner(nn.Module):
    def __init__(self, esm_dim=2560, go_emb_dim=768, joint_dim=512):
        super().__init__()

        # Branch 1: Project Protein (ESM2)
        self.prot_projector = nn.Sequential(
            nn.Linear(esm_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, joint_dim),
            # No final activation to keep real values for dot product
        )

        # Branch 2: Project GO Label (from BioBERT/Node2Vec)
        # If we want to freeze GO embeddings, we can do it outside or not update this part if it was an embedding layer
        # Here we assume input is already an embedding vector
        self.go_projector = nn.Sequential(nn.Linear(go_emb_dim, joint_dim))

    def forward(self, esm_embeddings, go_embeddings):
        """
        esm_embeddings: [Batch_Size, esm_dim] (e.g. 2560)
        go_embeddings:  [Num_GO_Terms, go_emb_dim] (e.g. 768) - Can be full set or batch
        """
        # 1. Project both to joint space
        prot_vec = self.prot_projector(esm_embeddings)  # [Batch, joint_dim]
        go_vec = self.go_projector(go_embeddings)  # [Num_GO, joint_dim]

        # 2. Normalize vectors (Important for Cosine Similarity)
        prot_vec = F.normalize(prot_vec, p=2, dim=1)
        go_vec = F.normalize(go_vec, p=2, dim=1)

        # 3. Calculate Similarity (Alignment)
        # Result is matrix [Batch_Size, Num_GO_Terms]
        # Each position (i, j) is the probability (logit) that protein i has function j
        logits = torch.matmul(prot_vec, go_vec.T)

        return logits  # Pass through Sigmoid -> BCE Loss externally
