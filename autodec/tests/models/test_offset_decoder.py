import torch


def test_cross_attention_offset_decoder_returns_offsets_and_attention_weights():
    from autodec.models.offset_decoder import CrossAttentionOffsetDecoder

    decoder = CrossAttentionOffsetDecoder(
        point_in_dim=10,
        primitive_in_dim=8,
        hidden_dim=12,
        n_heads=3,
    )

    offsets, attn = decoder(
        point_features=torch.randn(2, 5, 10),
        primitive_tokens=torch.randn(2, 3, 8),
        return_attention=True,
    )

    assert offsets.shape == (2, 5, 3)
    assert attn.shape == (2, 5, 3)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_cross_attention_offset_decoder_runs_stacked_within_primitive_blocks():
    from autodec.models.offset_decoder import CrossAttentionOffsetDecoder

    decoder = CrossAttentionOffsetDecoder(
        point_in_dim=10,
        primitive_in_dim=8,
        hidden_dim=12,
        n_heads=3,
        n_blocks=2,
        self_attention_mode="within_primitive",
    )

    offsets, attn = decoder(
        point_features=torch.randn(2, 6, 10),
        primitive_tokens=torch.randn(2, 3, 8),
        return_attention=True,
    )

    assert len(decoder.blocks) == 2
    assert offsets.shape == (2, 6, 3)
    assert attn.shape == (2, 6, 3)


def test_build_offset_decoder_builds_cross_attention_decoder():
    from autodec.models.offset_decoder import CrossAttentionOffsetDecoder, build_offset_decoder

    decoder = build_offset_decoder(
        decoder_type="cross_attention",
        point_in_dim=10,
        primitive_in_dim=8,
        hidden_dim=12,
        n_heads=3,
        n_blocks=2,
        self_attention_mode="within_primitive",
    )

    assert isinstance(decoder, CrossAttentionOffsetDecoder)
    assert len(decoder.blocks) == 2
