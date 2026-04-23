import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        half = self.embedding_dim // 2
        device = t.device
        dtype = t.dtype
        freq = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = t.unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < self.embedding_dim:
            emb = torch.nn.functional.pad(emb, (0, self.embedding_dim - emb.shape[-1]))
        return self.net(emb)
