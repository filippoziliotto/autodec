import torch


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
