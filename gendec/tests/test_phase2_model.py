import torch


def test_joint_model_forward_returns_three_tensors():
    from gendec.models.set_transformer_flow import JointSetTransformerFlowModel

    model = JointSetTransformerFlowModel(
        explicit_dim=15, residual_dim=64, hidden_dim=32, n_blocks=2, n_heads=4
    )
    et = torch.randn(2, 16, 79)
    t = torch.rand(2)
    v_hat_e, v_hat_z, v_hat = model(et, t)

    assert v_hat_e.shape == (2, 16, 15)
    assert v_hat_z.shape == (2, 16, 64)
    assert v_hat.shape == (2, 16, 79)


def test_joint_model_concat_equals_cat_of_branches():
    from gendec.models.set_transformer_flow import JointSetTransformerFlowModel

    model = JointSetTransformerFlowModel(
        explicit_dim=15, residual_dim=64, hidden_dim=32, n_blocks=2, n_heads=4
    )
    et = torch.randn(1, 16, 79)
    t = torch.rand(1)
    v_hat_e, v_hat_z, v_hat = model(et, t)

    assert torch.allclose(v_hat, torch.cat([v_hat_e, v_hat_z], dim=-1))


def test_joint_model_custom_residual_dim():
    from gendec.models.set_transformer_flow import JointSetTransformerFlowModel

    model = JointSetTransformerFlowModel(
        explicit_dim=15, residual_dim=32, hidden_dim=32, n_blocks=2, n_heads=4
    )
    et = torch.randn(1, 16, 47)
    t = torch.rand(1)
    v_hat_e, v_hat_z, v_hat = model(et, t)

    assert v_hat_e.shape == (1, 16, 15)
    assert v_hat_z.shape == (1, 16, 32)
    assert v_hat.shape == (1, 16, 47)


def test_joint_model_is_differentiable():
    from gendec.models.set_transformer_flow import JointSetTransformerFlowModel

    model = JointSetTransformerFlowModel(
        explicit_dim=15, residual_dim=64, hidden_dim=32, n_blocks=2, n_heads=4
    )
    et = torch.randn(1, 16, 79, requires_grad=False)
    t = torch.rand(1)
    _, _, v_hat = model(et, t)
    loss = v_hat.sum()
    loss.backward()
    # If no exception, gradients flow correctly
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
