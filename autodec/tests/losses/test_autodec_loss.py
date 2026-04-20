import torch


class FixedAngleSampler:
    def sample_on_batch(self, scale, shape):
        batch, primitives = scale.shape[:2]
        etas = torch.zeros(batch, primitives, 1).numpy()
        omegas = torch.zeros(batch, primitives, 1).numpy()
        return etas, omegas


def _batch():
    return {
        "points": torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
    }


def _outdict():
    surface_points = torch.tensor([[[1.0, 0.0, 0.0]]])
    decoded_points = torch.tensor([[[1.0, 0.0, 0.0]]])
    return {
        "decoded_points": decoded_points,
        "decoded_weights": torch.ones(1, 1),
        "surface_points": surface_points,
        "decoded_offsets": torch.zeros(1, 1, 3),
        "scale": torch.ones(1, 1, 3),
        "shape": torch.ones(1, 1, 2),
        "rotate": torch.eye(3).view(1, 1, 3, 3),
        "trans": torch.zeros(1, 1, 3),
        "exist_logit": torch.ones(1, 1, 1) * 20.0,
        "exist": torch.ones(1, 1, 1),
        "assign_matrix": torch.ones(1, 2, 1),
    }


def test_autodec_loss_phase1_uses_reconstruction_only():
    from autodec.losses.autodec_loss import AutoDecLoss

    loss_fn = AutoDecLoss(
        phase=1,
        lambda_sq=10.0,
        lambda_par=10.0,
        lambda_exist=10.0,
        n_sq_samples=1,
        angle_sampler=FixedAngleSampler(),
    )

    loss, metrics = loss_fn(_batch(), _outdict())

    assert loss.item() == metrics["recon"]
    assert "sq_loss" not in metrics
    assert metrics["all"] == metrics["recon"]


def test_autodec_loss_is_exported_from_package_root():
    from autodec import AutoDecLoss

    assert AutoDecLoss.__name__ == "AutoDecLoss"


def test_autodec_loss_phase2_composes_regularizers_and_metrics():
    from autodec.losses.autodec_loss import AutoDecLoss

    loss_fn = AutoDecLoss(
        phase=2,
        lambda_sq=2.0,
        lambda_par=3.0,
        lambda_exist=5.0,
        exist_point_threshold=1.0,
        n_sq_samples=1,
        angle_sampler=FixedAngleSampler(),
    )

    loss, metrics = loss_fn(_batch(), _outdict())
    expected = (
        metrics["recon"]
        + 2.0 * metrics["sq_loss"]
        + 3.0 * metrics["parsimony_loss"]
        + 5.0 * metrics["exist_loss"]
    )

    assert abs(loss.item() - expected) < 1e-6
    assert metrics["offset_ratio"] == 0.0
    assert metrics["active_primitive_count"] == 1.0
    assert "primitive_mass_entropy" in metrics
    assert "scaffold_chamfer" in metrics


def test_autodec_loss_logs_gated_offset_ratio_and_cap_saturation():
    from autodec.losses.autodec_loss import AutoDecLoss

    batch = {"points": torch.zeros(1, 2, 3)}
    outdict = _outdict()
    outdict["decoded_points"] = torch.zeros(1, 2, 3)
    outdict["surface_points"] = torch.tensor([[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    outdict["decoded_offsets"] = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    outdict["decoded_weights"] = torch.tensor([[0.5, 0.25]])
    outdict["offset_limit"] = torch.tensor([[[4.0], [2.0]]])

    _, metrics = AutoDecLoss(phase=1)(batch, outdict)

    assert metrics["offset_ratio"] == 1.0
    assert abs(metrics["gated_offset_ratio"] - 0.375) < 1e-6
    assert abs(metrics["offset_cap_saturation"] - 0.25) < 1e-6
    assert abs(metrics["offset_cap_saturated_fraction"] - (1.0 / 6.0)) < 1e-6


def test_autodec_loss_consistency_uses_zero_residual_decoder_points():
    from autodec.losses.autodec_loss import AutoDecLoss

    batch = {"points": torch.tensor([[[0.0, 0.0, 0.0]]])}
    outdict = _outdict()
    outdict["decoded_points"] = torch.tensor([[[0.0, 0.0, 0.0]]])
    outdict["surface_points"] = torch.tensor([[[5.0, 0.0, 0.0]]])
    outdict["consistency_decoded_points"] = torch.tensor([[[2.0, 0.0, 0.0]]])

    loss_fn = AutoDecLoss(phase=1, lambda_cons=0.5)

    loss, metrics = loss_fn(batch, outdict)

    assert metrics["recon"] == 0.0
    assert metrics["scaffold_chamfer"] == 50.0
    assert metrics["consistency_loss"] == 8.0
    assert loss.item() == 4.0


def test_autodec_loss_requires_consistency_decoder_points_when_enabled():
    from autodec.losses.autodec_loss import AutoDecLoss

    loss_fn = AutoDecLoss(phase=1, lambda_cons=1.0)

    try:
        loss_fn(_batch(), _outdict())
    except ValueError as exc:
        assert "consistency_decoded_points" in str(exc)
    else:
        raise AssertionError("Expected missing consistency decoder output to fail")
