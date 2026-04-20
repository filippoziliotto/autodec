import torch
import torch.nn as nn


class FixedAngleSampler:
    def sample_on_batch(self, scale, shape):
        batch, primitives = scale.shape[:2]
        etas = torch.zeros(batch, primitives, 2).numpy()
        omegas = torch.tensor([0.0, torch.pi / 2]).view(1, 1, 2).repeat(
            batch, primitives, 1
        ).numpy()
        return etas, omegas


def _encoder_outdict(batch=1, primitives=2, residual_dim=4):
    return {
        "scale": torch.ones(batch, primitives, 3),
        "shape": torch.ones(batch, primitives, 2),
        "trans": torch.zeros(batch, primitives, 3),
        "rotate": torch.eye(3).view(1, 1, 3, 3).repeat(batch, primitives, 1, 1),
        "exist_logit": torch.zeros(batch, primitives, 1),
        "exist": torch.ones(batch, primitives, 1) * 0.5,
        "residual": torch.randn(batch, primitives, residual_dim),
    }


def test_autodec_decoder_builds_features_and_gates_offsets():
    from autodec.decoder import AutoDecDecoder

    decoder = AutoDecDecoder(
        residual_dim=4,
        n_surface_samples=2,
        hidden_dim=16,
        n_heads=4,
        positional_frequencies=2,
        component_feature_dim=4,
        n_blocks=2,
        self_attention_mode="within_primitive",
        angle_sampler=FixedAngleSampler(),
    )
    for param in decoder.offset_decoder.parameters():
        torch.nn.init.zeros_(param)

    out = decoder(_encoder_outdict())

    assert out["surface_position_features"].shape == (1, 4, 15)
    assert torch.allclose(out["surface_position_features"][..., :3], out["surface_points"])
    assert out["decoder_features"].shape == (1, 4, 16)
    assert out["primitive_tokens"].shape == (1, 2, 8)
    assert out["projected_E_dec"].shape == (1, 2, 4)
    assert out["projected_residual"].shape == (1, 2, 4)
    assert out["projected_gates"].shape == (1, 4, 4)
    assert out["decoded_offsets"].shape == (1, 4, 3)
    assert out["decoded_points"].shape == (1, 4, 3)
    assert torch.allclose(out["decoded_points"], out["surface_points"])
    assert torch.allclose(out["decoded_weights"], torch.ones(1, 4) * 0.5)


def test_autodec_decoder_can_disable_positional_encoding_for_checkpoint_compatibility():
    from autodec.decoder import AutoDecDecoder

    decoder = AutoDecDecoder(
        residual_dim=4,
        n_surface_samples=2,
        hidden_dim=16,
        n_heads=4,
        positional_frequencies=0,
        component_feature_dim=0,
        n_blocks=1,
        self_attention_mode="none",
        angle_sampler=FixedAngleSampler(),
    )
    for param in decoder.offset_decoder.parameters():
        torch.nn.init.zeros_(param)

    out = decoder(_encoder_outdict())

    assert out["surface_position_features"].shape == (1, 4, 3)
    assert out["decoder_features"].shape == (1, 4, 26)


class ResidualEchoOffsetDecoder(nn.Module):
    def __init__(self, residual_dim):
        super().__init__()
        self.residual_dim = residual_dim

    def forward(self, point_features, primitive_tokens, return_attention=False):
        residual = point_features[..., -(self.residual_dim + 1) : -1]
        offsets = point_features.new_zeros(point_features.shape[0], point_features.shape[1], 3)
        offsets[..., 0] = residual.mean(dim=-1)
        if return_attention:
            attention = point_features.new_ones(point_features.shape[0], point_features.shape[1], 1)
            return offsets, attention
        return offsets


class ConstantOffsetDecoder(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, point_features, primitive_tokens, return_attention=False):
        offsets = point_features.new_full(
            (point_features.shape[0], point_features.shape[1], 3),
            self.value,
        )
        if return_attention:
            attention = point_features.new_ones(
                point_features.shape[0],
                point_features.shape[1],
                1,
            )
            return offsets, attention
        return offsets


def test_autodec_decoder_return_consistency_uses_zero_residual_pass():
    from autodec.decoder import AutoDecDecoder

    residual_dim = 4
    decoder = AutoDecDecoder(
        residual_dim=residual_dim,
        n_surface_samples=2,
        hidden_dim=16,
        n_heads=4,
        positional_frequencies=0,
        component_feature_dim=0,
        n_blocks=1,
        self_attention_mode="none",
        angle_sampler=FixedAngleSampler(),
    )
    decoder.offset_decoder = ResidualEchoOffsetDecoder(residual_dim)
    outdict = _encoder_outdict(residual_dim=residual_dim)
    outdict["residual"] = torch.ones_like(outdict["residual"])

    out = decoder(outdict, return_consistency=True)

    assert torch.allclose(out["decoded_offsets"][..., 0], torch.ones(1, 4))
    assert torch.allclose(out["consistency_decoded_offsets"][..., 0], torch.zeros(1, 4))
    assert torch.allclose(out["consistency_decoded_points"], out["surface_points"])


def test_autodec_decoder_scale_caps_offsets_per_primitive():
    from autodec.decoder import AutoDecDecoder

    decoder = AutoDecDecoder(
        residual_dim=4,
        n_surface_samples=2,
        hidden_dim=16,
        n_heads=4,
        positional_frequencies=0,
        component_feature_dim=0,
        n_blocks=1,
        self_attention_mode="none",
        offset_cap=0.3,
        angle_sampler=FixedAngleSampler(),
    )
    decoder.offset_decoder = ConstantOffsetDecoder(10.0)
    outdict = _encoder_outdict()
    outdict["scale"] = torch.tensor([[[0.3, 0.6, 0.9], [0.03, 0.06, 0.09]]])

    out = decoder(outdict)

    expected_caps = torch.tensor([0.18, 0.18, 0.018, 0.018]).view(1, 4, 1)
    assert torch.allclose(out["offset_limit"], expected_caps)
    assert torch.allclose(out["decoded_offsets"], expected_caps.expand_as(out["decoded_offsets"]))
    assert torch.allclose(
        out["decoded_points"],
        out["surface_points"] + out["decoded_weights"].unsqueeze(-1) * out["decoded_offsets"],
    )


def test_autodec_decoder_offset_cap_none_preserves_raw_offsets():
    from autodec.decoder import AutoDecDecoder

    decoder = AutoDecDecoder(
        residual_dim=4,
        n_surface_samples=2,
        hidden_dim=16,
        n_heads=4,
        positional_frequencies=0,
        component_feature_dim=0,
        n_blocks=1,
        self_attention_mode="none",
        offset_cap=None,
        angle_sampler=FixedAngleSampler(),
    )
    decoder.offset_decoder = ConstantOffsetDecoder(10.0)

    out = decoder(_encoder_outdict())

    assert torch.allclose(out["decoded_offsets"], torch.full_like(out["decoded_offsets"], 10.0))


def test_autodec_decoder_offset_cap_does_not_backprop_into_scale():
    from autodec.decoder import AutoDecDecoder

    decoder = AutoDecDecoder(
        residual_dim=4,
        n_surface_samples=2,
        hidden_dim=16,
        n_heads=4,
        positional_frequencies=0,
        component_feature_dim=0,
        n_blocks=1,
        self_attention_mode="none",
        offset_cap=0.3,
        angle_sampler=FixedAngleSampler(),
    )
    offsets = torch.ones(1, 4, 3, requires_grad=True)
    scale = torch.ones(1, 2, 3, requires_grad=True)
    part_ids = torch.tensor([0, 0, 1, 1])

    capped_offsets, _ = decoder._apply_offset_cap(offsets, scale, part_ids)
    capped_offsets.sum().backward()

    assert offsets.grad is not None
    assert scale.grad is None
