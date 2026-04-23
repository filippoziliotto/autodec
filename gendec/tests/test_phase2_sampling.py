import torch


def _make_joint_model(explicit_dim=15, residual_dim=64):
    from gendec.models.set_transformer_flow import JointSetTransformerFlowModel

    return JointSetTransformerFlowModel(
        explicit_dim=explicit_dim,
        residual_dim=residual_dim,
        hidden_dim=32,
        n_blocks=2,
        n_heads=4,
    )


def _dummy_stats(token_dim=79):
    return {"mean": torch.zeros(token_dim), "std": torch.ones(token_dim)}


def test_euler_sample_joint_shape():
    from gendec.sampling import euler_sample_joint

    model = _make_joint_model()
    tokens = euler_sample_joint(model, num_samples=3, token_dim=79, num_steps=5)
    assert tokens.shape == (3, 16, 79)


def test_postprocess_joint_tokens_splits_correctly():
    from gendec.sampling import postprocess_joint_tokens

    tokens_ez = torch.zeros(2, 16, 79)
    stats = _dummy_stats(79)
    processed = postprocess_joint_tokens(tokens_ez, stats, explicit_dim=15)

    assert "scale" in processed
    assert "shape" in processed
    assert "rotate" in processed
    assert "trans" in processed
    assert "exist" in processed
    assert "active_mask" in processed
    assert "tokens" in processed
    assert "tokens_z" in processed
    assert "tokens_ez" in processed
    assert processed["tokens"].shape == (2, 16, 15)
    assert processed["tokens_z"].shape == (2, 16, 64)
    assert processed["tokens_ez"].shape == (2, 16, 79)


def test_postprocess_joint_tokens_active_mask_dtype():
    from gendec.sampling import postprocess_joint_tokens

    tokens_ez = torch.zeros(2, 16, 79)
    stats = _dummy_stats(79)
    processed = postprocess_joint_tokens(tokens_ez, stats)
    assert processed["active_mask"].dtype == torch.bool


def test_sample_joint_scaffolds_output_keys():
    from gendec.sampling import sample_joint_scaffolds

    model = _make_joint_model()
    stats = _dummy_stats(79)
    processed = sample_joint_scaffolds(model, stats, num_samples=2, token_dim=79, num_steps=5)

    assert "preview_points" in processed
    assert "tokens_z" in processed
    assert processed["tokens_z"].shape[0] == 2
    assert processed["tokens_z"].shape[-1] == 64


def test_sample_joint_scaffolds_explicit_dim_respected():
    from gendec.sampling import sample_joint_scaffolds

    model = _make_joint_model(explicit_dim=15, residual_dim=32)
    stats = _dummy_stats(47)
    processed = sample_joint_scaffolds(
        model, stats, num_samples=1, token_dim=47, num_steps=3, explicit_dim=15
    )
    assert processed["tokens"].shape == (1, 16, 15)
    assert processed["tokens_z"].shape == (1, 16, 32)
