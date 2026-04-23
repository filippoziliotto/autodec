from types import SimpleNamespace

import torch


def test_build_flow_batch_constructs_interpolation_and_target_velocity():
    from gendec.losses.flow_matching import build_flow_batch

    e0 = torch.zeros(2, 16, 15)
    e1 = torch.ones(2, 16, 15)
    t = torch.tensor([0.25, 0.75])

    batch = build_flow_batch(e0, e1=e1, t=t)

    assert batch["Et"].shape == (2, 16, 15)
    assert batch["velocity_target"].shape == (2, 16, 15)
    assert torch.allclose(batch["Et"][0], torch.full((16, 15), 0.25))
    assert torch.allclose(batch["Et"][1], torch.full((16, 15), 0.75))
    assert torch.allclose(batch["velocity_target"], torch.ones(2, 16, 15))


def test_flow_matching_loss_returns_flow_and_optional_existence_metrics():
    from gendec.losses.flow_matching import FlowMatchingLoss

    loss_fn = FlowMatchingLoss(lambda_exist=0.05, exist_channel=-1)
    e0 = torch.zeros(1, 16, 15)
    e1 = torch.ones(1, 16, 15)
    t = torch.tensor([0.5])
    v_hat = torch.ones(1, 16, 15)

    loss, metrics = loss_fn(
        {
            "E0": e0,
            "E1": e1,
            "Et": torch.full((1, 16, 15), 0.5),
            "velocity_target": e1 - e0,
            "t": t,
            "exist": torch.ones(1, 16, 1),
        },
        v_hat,
    )

    assert torch.is_tensor(loss)
    assert set(metrics) >= {"flow_loss", "all"}
