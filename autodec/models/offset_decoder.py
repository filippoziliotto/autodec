import torch.nn as nn


class OffsetDecoderBlock(nn.Module):
    """Self-attention over sampled points followed by primitive cross-attention."""

    def __init__(
        self,
        hidden_dim,
        n_heads=4,
        self_attention_mode="none",
        dim_feedforward=None,
    ):
        super().__init__()
        if self_attention_mode not in {"none", "within_primitive"}:
            raise ValueError(f"Unsupported self_attention_mode: {self_attention_mode}")
        self.self_attention_mode = self_attention_mode
        self.self_attention = None
        if self_attention_mode == "within_primitive":
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                batch_first=True,
            )
            self.self_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        dim_feedforward = dim_feedforward or hidden_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def _within_primitive_self_attention(self, point_hidden, n_primitives):
        batch, n_points, hidden_dim = point_hidden.shape
        if n_points % n_primitives != 0:
            raise ValueError(
                "within_primitive self-attention requires primitive-major points "
                "with the same number of samples per primitive"
            )
        n_samples = n_points // n_primitives
        grouped = point_hidden.reshape(batch, n_primitives, n_samples, hidden_dim)
        grouped = grouped.reshape(batch * n_primitives, n_samples, hidden_dim)
        attended, _ = self.self_attention(
            query=grouped,
            key=grouped,
            value=grouped,
            need_weights=False,
        )
        attended = attended.reshape(batch, n_primitives, n_samples, hidden_dim)
        return attended.reshape(batch, n_points, hidden_dim)

    def forward(self, point_hidden, primitive_hidden, return_attention=False):
        if self.self_attention is not None:
            attended = self._within_primitive_self_attention(
                point_hidden,
                primitive_hidden.shape[1],
            )
            point_hidden = self.self_norm(point_hidden + attended)

        attended, attention_weights = self.cross_attention(
            query=point_hidden,
            key=primitive_hidden,
            value=primitive_hidden,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        point_hidden = self.cross_norm(point_hidden + attended)
        point_hidden = self.ffn_norm(point_hidden + self.ffn(point_hidden))
        return point_hidden, attention_weights


class CrossAttentionOffsetDecoder(nn.Module):
    """Option B decoder with stacked point and primitive attention blocks."""

    def __init__(
        self,
        point_in_dim,
        primitive_in_dim,
        hidden_dim=128,
        n_heads=4,
        offset_scale=None,
        n_blocks=1,
        self_attention_mode="none",
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        self.point_in_dim = point_in_dim
        self.primitive_in_dim = primitive_in_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.offset_scale = offset_scale
        self.n_blocks = n_blocks
        self.self_attention_mode = self_attention_mode

        self.point_proj = nn.Linear(point_in_dim, hidden_dim)
        self.primitive_proj = nn.Linear(primitive_in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                OffsetDecoderBlock(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    self_attention_mode=self_attention_mode,
                )
                for _ in range(n_blocks)
            ]
        )
        self.offset_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, point_features, primitive_tokens, return_attention=False):
        point_hidden = self.point_proj(point_features)
        primitive_hidden = self.primitive_proj(primitive_tokens)
        attention_weights = None
        for block in self.blocks:
            point_hidden, attention_weights = block(
                point_hidden,
                primitive_hidden,
                return_attention=return_attention,
            )
        offsets = self.offset_mlp(point_hidden)
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
    n_blocks=1,
    self_attention_mode="none",
):
    if decoder_type == "cross_attention":
        return CrossAttentionOffsetDecoder(
            point_in_dim=point_in_dim,
            primitive_in_dim=primitive_in_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            offset_scale=offset_scale,
            n_blocks=n_blocks,
            self_attention_mode=self_attention_mode,
        )
    raise ValueError(f"Unsupported offset decoder type: {decoder_type}")
