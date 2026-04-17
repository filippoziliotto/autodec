import torch


def test_weighted_chamfer_forward_does_not_let_inactive_outlier_dominate():
    from autodec.losses.chamfer import weighted_chamfer_l2

    pred = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0]]])
    weights = torch.tensor([[1.0, 0.0]])

    loss, components = weighted_chamfer_l2(pred, target, weights, return_components=True)

    assert loss.item() < 1e-3
    assert components["forward"].item() < 1e-3
    assert components["backward"].item() == 0.0


def test_weighted_chamfer_backward_discourages_low_weight_matches():
    from autodec.losses.chamfer import weighted_chamfer_l2

    pred = torch.tensor([[[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0]]])
    weights = torch.tensor([[0.001, 1.0]])

    _, components = weighted_chamfer_l2(
        pred,
        target,
        weights,
        min_backward_weight=0.001,
        return_components=True,
    )

    assert torch.allclose(components["backward"], torch.tensor(0.25))
