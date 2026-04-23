import torch
import torch.nn as nn

from gendec.models.components import ClassConditioning, GlobalToken, SetTransformerBlock, TokenProjection, VelocityHead
from gendec.models.time_embedding import SinusoidalTimeEmbedding
from gendec.tokens import TOKEN_DIM


class SetTransformerFlowModel(nn.Module):
    def __init__(
        self,
        token_dim=15,
        hidden_dim=256,
        n_blocks=6,
        n_heads=8,
        dropout=0.0,
        conditioning_enabled=False,
        num_classes=1,
        class_embed_dim=None,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_classes = int(num_classes)
        self.conditioning_enabled = bool(conditioning_enabled)
        self.conditioning_active = self.conditioning_enabled and self.num_classes > 1
        self.token_projection = TokenProjection(token_dim=token_dim, hidden_dim=hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim=hidden_dim)
        self.global_token = GlobalToken(hidden_dim=hidden_dim)
        self.class_conditioning = (
            ClassConditioning(self.num_classes, hidden_dim=hidden_dim, embedding_dim=class_embed_dim)
            if self.conditioning_active
            else None
        )
        self.blocks = nn.ModuleList(
            [SetTransformerBlock(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)]
        )
        self.velocity_head = VelocityHead(hidden_dim=hidden_dim, token_dim=token_dim)

    def _class_hidden(self, category_index, batch_size, device):
        if not self.conditioning_active:
            return None
        if category_index is None:
            category_index = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif not torch.is_tensor(category_index):
            category_index = torch.tensor(category_index, dtype=torch.long, device=device)
        else:
            category_index = category_index.to(device=device, dtype=torch.long).view(batch_size)
        return self.class_conditioning(category_index)

    def forward(self, et, t, category_index=None):
        token_hidden = self.token_projection(et)
        time_hidden = self.time_embedding(t).unsqueeze(1)
        token_hidden = token_hidden + time_hidden
        class_hidden = self._class_hidden(category_index, et.shape[0], et.device)
        if class_hidden is not None:
            token_hidden = token_hidden + class_hidden.unsqueeze(1)
        global_token = self.global_token(et.shape[0])
        if class_hidden is not None:
            global_token = global_token + class_hidden.unsqueeze(1)
        hidden = torch.cat([global_token, token_hidden], dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.velocity_head(hidden[:, 1:, :])


class JointSetTransformerFlowModel(nn.Module):
    """Phase 2 flow model for joint (E, Z) tokens.

    Uses a shared Set Transformer backbone with two separate output heads:
      - ``explicit_head``: predicts velocity for the 15D explicit scaffold tokens
      - ``residual_head``: predicts velocity for the residual latent tokens

    The forward pass returns a tuple ``(v_hat_e, v_hat_z, v_hat)`` where
    ``v_hat = cat([v_hat_e, v_hat_z], dim=-1)`` for use in the full-token loss.

    Args:
        explicit_dim: dimension of explicit scaffold tokens (default 15)
        residual_dim: dimension of residual latent tokens (default 64)
        hidden_dim:   transformer hidden dimension (default 384)
        n_blocks:     number of SetTransformerBlocks (default 6)
        n_heads:      number of attention heads (default 8)
        dropout:      attention dropout (default 0.0)
    """

    def __init__(
        self,
        explicit_dim=TOKEN_DIM,
        residual_dim=64,
        hidden_dim=384,
        n_blocks=6,
        n_heads=8,
        dropout=0.0,
        conditioning_enabled=False,
        num_classes=1,
        class_embed_dim=None,
    ):
        super().__init__()
        self.explicit_dim = int(explicit_dim)
        self.residual_dim = int(residual_dim)
        self.token_dim = self.explicit_dim + self.residual_dim
        self.num_classes = int(num_classes)
        self.conditioning_enabled = bool(conditioning_enabled)
        self.conditioning_active = self.conditioning_enabled and self.num_classes > 1

        self.token_projection = TokenProjection(token_dim=self.token_dim, hidden_dim=hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim=hidden_dim)
        self.global_token = GlobalToken(hidden_dim=hidden_dim)
        self.class_conditioning = (
            ClassConditioning(self.num_classes, hidden_dim=hidden_dim, embedding_dim=class_embed_dim)
            if self.conditioning_active
            else None
        )
        self.blocks = nn.ModuleList(
            [SetTransformerBlock(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)]
        )
        self.explicit_head = VelocityHead(hidden_dim=hidden_dim, token_dim=self.explicit_dim)
        self.residual_head = VelocityHead(hidden_dim=hidden_dim, token_dim=self.residual_dim)

    def _class_hidden(self, category_index, batch_size, device):
        if not self.conditioning_active:
            return None
        if category_index is None:
            category_index = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif not torch.is_tensor(category_index):
            category_index = torch.tensor(category_index, dtype=torch.long, device=device)
        else:
            category_index = category_index.to(device=device, dtype=torch.long).view(batch_size)
        return self.class_conditioning(category_index)

    def forward(self, tt, t, category_index=None):
        """Forward pass.

        Args:
            tt: joint noisy tokens [B, 16, explicit_dim + residual_dim]
            t:  time scalars [B]

        Returns:
            v_hat_e: [B, 16, explicit_dim]
            v_hat_z: [B, 16, residual_dim]
            v_hat:   [B, 16, explicit_dim + residual_dim]
        """
        token_hidden = self.token_projection(tt)
        time_hidden = self.time_embedding(t).unsqueeze(1)
        token_hidden = token_hidden + time_hidden
        class_hidden = self._class_hidden(category_index, tt.shape[0], tt.device)
        if class_hidden is not None:
            token_hidden = token_hidden + class_hidden.unsqueeze(1)
        global_token = self.global_token(tt.shape[0])
        if class_hidden is not None:
            global_token = global_token + class_hidden.unsqueeze(1)
        hidden = torch.cat([global_token, token_hidden], dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        prim_hidden = hidden[:, 1:, :]
        v_hat_e = self.explicit_head(prim_hidden)
        v_hat_z = self.residual_head(prim_hidden)
        v_hat = torch.cat([v_hat_e, v_hat_z], dim=-1)
        return v_hat_e, v_hat_z, v_hat
