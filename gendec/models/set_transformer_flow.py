import torch
import torch.nn as nn

from gendec.models.components import GlobalToken, SetTransformerBlock, TokenProjection, VelocityHead
from gendec.models.time_embedding import SinusoidalTimeEmbedding


class SetTransformerFlowModel(nn.Module):
    def __init__(
        self,
        token_dim=15,
        hidden_dim=256,
        n_blocks=6,
        n_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.token_projection = TokenProjection(token_dim=token_dim, hidden_dim=hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim=hidden_dim)
        self.global_token = GlobalToken(hidden_dim=hidden_dim)
        self.blocks = nn.ModuleList(
            [SetTransformerBlock(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)]
        )
        self.velocity_head = VelocityHead(hidden_dim=hidden_dim, token_dim=token_dim)

    def forward(self, et, t):
        token_hidden = self.token_projection(et)
        time_hidden = self.time_embedding(t).unsqueeze(1)
        token_hidden = token_hidden + time_hidden
        global_token = self.global_token(et.shape[0])
        hidden = torch.cat([global_token, token_hidden], dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.velocity_head(hidden[:, 1:, :])
