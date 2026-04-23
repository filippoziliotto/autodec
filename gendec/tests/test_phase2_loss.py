import torch


def _make_joint_batch(B=2, N=16, explicit_dim=15, residual_dim=64):
    token_dim = explicit_dim + residual_dim
    e0 = torch.zeros(B, N, token_dim)
    e1 = torch.ones(B, N, token_dim)
    t = torch.full((B,), 0.5)
    et = e0 + t.view(B, 1, 1) * (e1 - e0)
    return {
        "E0": e0,
        "E1": e1,
        "Et": et,
        "velocity_target": e1 - e0,
        "t": t,
        "exist": torch.ones(B, N, 1),
    }


def test_joint_loss_returns_scalar_and_metrics():
    from gendec.losses.flow_matching import JointFlowMatchingLoss

    loss_fn = JointFlowMatchingLoss(explicit_dim=15, lambda_e=1.0, lambda_z=1.0, lambda_exist=0.0)
    batch = _make_joint_batch()
    v_hat_e = torch.ones(2, 16, 15)
    v_hat_z = torch.ones(2, 16, 64)

    loss, metrics = loss_fn(batch, v_hat_e, v_hat_z)

    assert loss.shape == ()
    assert "flow_loss_e" in metrics
    assert "flow_loss_z" in metrics
    assert "flow_loss" in metrics
    assert "all" in metrics


def test_joint_loss_zero_for_perfect_prediction():
    from gendec.losses.flow_matching import JointFlowMatchingLoss

    loss_fn = JointFlowMatchingLoss(explicit_dim=15, lambda_e=1.0, lambda_z=1.0, lambda_exist=0.0)
    batch = _make_joint_batch()
    # Perfect prediction = velocity_target
    v_hat_e = torch.ones(2, 16, 15)
    v_hat_z = torch.ones(2, 16, 64)

    loss, metrics = loss_fn(batch, v_hat_e, v_hat_z)
    assert abs(float(loss)) < 1e-6


def test_joint_loss_with_existence_aux():
    from gendec.losses.flow_matching import JointFlowMatchingLoss

    loss_fn = JointFlowMatchingLoss(explicit_dim=15, lambda_e=1.0, lambda_z=1.0, lambda_exist=0.05, exist_channel=-1)
    batch = _make_joint_batch()
    batch["token_mean"] = torch.zeros(79)
    batch["token_std"] = torch.ones(79)
    v_hat_e = torch.zeros(2, 16, 15)
    v_hat_z = torch.zeros(2, 16, 64)

    loss, metrics = loss_fn(batch, v_hat_e, v_hat_z)
    assert "exist_loss" in metrics
    assert torch.is_tensor(loss)


def test_joint_loss_separates_explicit_and_residual_branches():
    from gendec.losses.flow_matching import JointFlowMatchingLoss

    loss_fn = JointFlowMatchingLoss(explicit_dim=15, lambda_e=1.0, lambda_z=0.0, lambda_exist=0.0)
    batch = _make_joint_batch()
    # Perfect explicit, imperfect residual
    v_hat_e = torch.ones(2, 16, 15)
    v_hat_z = torch.zeros(2, 16, 64)  # wrong

    loss, metrics = loss_fn(batch, v_hat_e, v_hat_z)
    # lambda_z=0 so residual error contributes 0 to total
    assert abs(metrics["flow_loss_e"]) < 1e-6
    assert metrics["flow_loss_z"] > 0.0
    assert abs(float(loss)) < 1e-6  # total is 0 since lambda_z=0


def test_joint_loss_per_sample_flag():
    from gendec.losses.flow_matching import JointFlowMatchingLoss

    loss_fn = JointFlowMatchingLoss(explicit_dim=15)
    batch = _make_joint_batch()
    v_hat_e = torch.zeros(2, 16, 15)
    v_hat_z = torch.zeros(2, 16, 64)

    _, metrics = loss_fn(batch, v_hat_e, v_hat_z, return_per_sample=True)
    assert "per_sample" in metrics
    assert "all" in metrics["per_sample"]
    assert metrics["per_sample"]["all"].shape == (2,)
