import torch


class FixedAngleSampler:
    def sample_on_batch(self, scale, shape):
        batch, primitives = scale.shape[:2]
        etas = torch.zeros(batch, primitives, 1).numpy()
        omegas = torch.zeros(batch, primitives, 1).numpy()
        return etas, omegas


def _sq_outdict():
    return {
        "scale": torch.ones(1, 1, 3),
        "shape": torch.ones(1, 1, 2),
        "rotate": torch.eye(3).view(1, 1, 3, 3),
        "trans": torch.zeros(1, 1, 3),
        "exist_logit": torch.ones(1, 1, 1) * 20.0,
        "exist": torch.ones(1, 1, 1),
        "assign_matrix": torch.ones(1, 2, 1),
    }


def test_sq_regularizer_uses_sampled_surface_chamfer_l2():
    from autodec.losses.sq_regularizer import SQRegularizer

    batch = {
        "points": torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
    }
    regularizer = SQRegularizer(n_samples=1, angle_sampler=FixedAngleSampler())

    loss, components = regularizer(batch, _sq_outdict(), return_components=True)

    assert torch.allclose(components["point_to_sq"], torch.tensor(0.5))
    assert torch.allclose(components["sq_to_point"], torch.tensor(0.0))
    assert torch.allclose(loss, torch.tensor(0.5))


def test_assignment_parsimony_prefers_concentrated_mass():
    from autodec.losses.sq_regularizer import assignment_parsimony_loss

    concentrated = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    balanced = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])

    assert assignment_parsimony_loss(concentrated) < assignment_parsimony_loss(balanced)


def test_existence_loss_uses_assignment_mass_targets():
    from autodec.losses.sq_regularizer import existence_loss

    assign_matrix = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    good_logits = torch.tensor([[[20.0], [-20.0]]])
    bad_logits = -good_logits

    good = existence_loss(
        assign_matrix,
        exist_logit=good_logits,
        point_threshold=2.0,
    )
    bad = existence_loss(
        assign_matrix,
        exist_logit=bad_logits,
        point_threshold=2.0,
    )

    assert good < bad
