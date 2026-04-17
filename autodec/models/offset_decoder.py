import torch.nn as nn


class CrossAttentionOffsetDecoder(nn.Module):
    """Option B decoder: surface point MLP with primitive-token cross-attention."""

    def __init__(
        self,
        point_in_dim,
        primitive_in_dim,
        hidden_dim=128,
        n_heads=4,
        offset_scale=None,
    ):
        super().__init__()
        self.point_in_dim = point_in_dim
        self.primitive_in_dim = primitive_in_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.offset_scale = offset_scale

        self.point_proj = nn.Linear(point_in_dim, hidden_dim)
        self.primitive_proj = nn.Linear(primitive_in_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.offset_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, point_features, primitive_tokens, return_attention=False):
        point_hidden = self.point_proj(point_features)
        primitive_hidden = self.primitive_proj(primitive_tokens)
        attended, attention_weights = self.cross_attention(
            query=point_hidden,
            key=primitive_hidden,
            value=primitive_hidden,
            need_weights=True,
            average_attn_weights=True,
        )
        offsets = self.offset_mlp(point_hidden + attended)
        if self.offset_scale is not None:
            offsets = self.offset_scale * offsets.tanh()
        if return_attention:
            return offsets, attention_weights
        return offsets


def build_offset_decoder(
    decoder_type,
    point_in_dim,
    primitive_in_dim,
    hidden_dim=128,
    n_heads=4,
    offset_scale=None,
):
    if decoder_type == "cross_attention":
        return CrossAttentionOffsetDecoder(
            point_in_dim=point_in_dim,
            primitive_in_dim=primitive_in_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            offset_scale=offset_scale,
        )
    raise ValueError(f"Unsupported offset decoder type: {decoder_type}")
