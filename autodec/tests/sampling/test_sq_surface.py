import torch


class FixedAngleSampler:
    def __init__(self, etas, omegas):
        self.etas = etas
        self.omegas = omegas

    def sample_on_batch(self, scale, shape):
        batch, primitives = scale.shape[:2]
        etas = self.etas.view(1, 1, -1).repeat(batch, primitives, 1).numpy()
        omegas = self.omegas.view(1, 1, -1).repeat(batch, primitives, 1).numpy()
        return etas, omegas


def _outdict(exist_logit=-4.0):
    return {
        "scale": torch.ones(1, 2, 3, requires_grad=True),
        "shape": torch.ones(1, 2, 2, requires_grad=True),
        "rotate": torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1),
        "trans": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], requires_grad=True),
        "exist_logit": torch.ones(1, 2, 1) * exist_logit,
    }


def test_sq_surface_sampler_returns_points_weights_and_part_ids():
    from autodec.sampling.sq_surface import SQSurfaceSampler

    sampler = SQSurfaceSampler(
        n_samples=2,
        tau=2.0,
        angle_sampler=FixedAngleSampler(
            etas=torch.tensor([0.0, 0.0]),
            omegas=torch.tensor([0.0, torch.pi / 2]),
        ),
    )

    sample = sampler(_outdict())

    assert sample.flat_points.shape == (1, 4, 3)
    assert sample.surface_points.shape == (1, 2, 2, 3)
    assert sample.weights.shape == (1, 4)
    assert torch.equal(sample.part_ids, torch.tensor([0, 0, 1, 1]))
    assert torch.allclose(sample.weights, torch.sigmoid(torch.ones(1, 4) * -2.0))


def test_sq_surface_sampler_does_not_scale_coordinates_by_existence_weight():
    from autodec.sampling.sq_surface import SQSurfaceSampler

    sampler = SQSurfaceSampler(
        n_samples=1,
        angle_sampler=FixedAngleSampler(
            etas=torch.tensor([0.0]),
            omegas=torch.tensor([0.0]),
        ),
    )

    sample = sampler(_outdict(exist_logit=-20.0))

    assert torch.allclose(sample.flat_points[0, 0], torch.tensor([2.0, 0.0, 0.0]))
    assert sample.weights[0, 0] < 1e-6


def test_sq_surface_sampler_clamps_out_of_range_shape_exponents():
    from autodec.sampling.sq_surface import SQSurfaceSampler

    eta = torch.tensor([torch.pi / 6])
    omega = torch.tensor([torch.pi / 3])
    outdict = {
        "scale": torch.ones(1, 1, 3),
        "shape": torch.tensor([[[0.0, 3.0]]]),
        "rotate": torch.eye(3).view(1, 1, 3, 3),
        "trans": torch.zeros(1, 1, 3),
        "exist_logit": torch.zeros(1, 1, 1),
    }
    sampler = SQSurfaceSampler(
        n_samples=1,
        angle_sampler=FixedAngleSampler(etas=eta, omegas=omega),
    )

    sample = sampler(outdict)

    expected = torch.tensor(
        [
            [
                [
                    torch.cos(eta)[0].pow(0.1) * torch.cos(omega)[0].pow(2.0),
                    torch.cos(eta)[0].pow(0.1) * torch.sin(omega)[0].pow(2.0),
                    torch.sin(eta)[0].pow(0.1),
                ]
            ]
        ]
    )
    assert torch.allclose(sample.flat_points, expected, atol=1e-6)


def test_sq_surface_sampler_keeps_gradients_to_sq_parameters():
    from autodec.sampling.sq_surface import SQSurfaceSampler

    outdict = _outdict()
    sampler = SQSurfaceSampler(
        n_samples=1,
        angle_sampler=FixedAngleSampler(
            etas=torch.tensor([0.0]),
            omegas=torch.tensor([0.0]),
        ),
    )

    sample = sampler(outdict)
    sample.flat_points.square().mean().backward()

    assert outdict["scale"].grad is not None
    assert outdict["shape"].grad is not None
    assert outdict["trans"].grad is not None
