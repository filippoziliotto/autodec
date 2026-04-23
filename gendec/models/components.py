import torch
import torch.nn as nn


class TokenProjection(nn.Module):
    def __init__(self, token_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tokens):
        return self.net(tokens)


class GlobalToken(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, batch_size):
        return self.token.expand(batch_size, -1, -1)


class VelocityHead(nn.Module):
    def __init__(self, hidden_dim, token_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, hidden):
        return self.net(hidden)


class SetTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden):
        attended, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        hidden = self.attn_norm(hidden + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        return hidden
