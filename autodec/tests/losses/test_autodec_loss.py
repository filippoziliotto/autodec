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
