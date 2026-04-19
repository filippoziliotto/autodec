from dataclasses import dataclass

import torch
import torch.nn as nn


MIN_SHAPE_EXPONENT = 0.1
MAX_SHAPE_EXPONENT = 2.0


@dataclass
class SQSurfaceSample:
    canonical_points: torch.Tensor
    surface_points: torch.Tensor
    flat_points: torch.Tensor
    part_ids: torch.Tensor
    weights: torch.Tensor


class SQSurfaceSampler(nn.Module):
    """Sample points on predicted superquadric surfaces."""

    def __init__(self, n_samples=256, tau=1.0, angle_sampler=None, eps=1e-6):
        super().__init__()
        self.n_samples = n_samples
        self.tau = tau
        self.angle_sampler = angle_sampler
        self.eps = eps

    def _angle_sampler(self):
        if self.angle_sampler is not None:
            return self.angle_sampler
        from superdec.loss.sampler import EqualDistanceSamplerSQ

        self.angle_sampler = EqualDistanceSamplerSQ(
            n_samples=self.n_samples,
            D_eta=0.05,
            D_omega=0.05,
        )
        return self.angle_sampler

    def _sample_angles(self, scale, shape):
        etas, omegas = self._angle_sampler().sample_on_batch(
            scale.detach().cpu().numpy(),
            shape.detach().cpu().numpy(),
        )
        etas = scale.new_tensor(etas)
        omegas = scale.new_tensor(omegas)
        return etas, omegas

    def _signed_power(self, value, exponent):
        return torch.sign(value) * torch.clamp(value.abs(), min=self.eps).pow(exponent)

    def _canonical_points(self, scale, shape, etas, omegas):
        shape = shape.clamp(MIN_SHAPE_EXPONENT, MAX_SHAPE_EXPONENT)
        e1 = shape[..., 0].unsqueeze(-1)
        e2 = shape[..., 1].unsqueeze(-1)
        sx = scale[..., 0].unsqueeze(-1)
        sy = scale[..., 1].unsqueeze(-1)
        sz = scale[..., 2].unsqueeze(-1)

        cos_eta = torch.cos(etas)
        sin_eta = torch.sin(etas)
        cos_omega = torch.cos(omegas)
        sin_omega = torch.sin(omegas)

        x = sx * self._signed_power(cos_eta, e1) * self._signed_power(cos_omega, e2)
        y = sy * self._signed_power(cos_eta, e1) * self._signed_power(sin_omega, e2)
        z = sz * self._signed_power(sin_eta, e1)
        return torch.stack([x, y, z], dim=-1)

    def forward(self, outdict):
        scale = outdict["scale"]
        shape = outdict["shape"].clamp(MIN_SHAPE_EXPONENT, MAX_SHAPE_EXPONENT)
        rotate = outdict["rotate"]
        trans = outdict["trans"]
        exist_logit = outdict["exist_logit"]

        etas, omegas = self._sample_angles(scale, shape)
        canonical = self._canonical_points(scale, shape, etas, omegas)
        surface = torch.matmul(rotate.unsqueeze(2), canonical.unsqueeze(-1)).squeeze(-1)
        surface = surface + trans.unsqueeze(2)

        batch, primitives, samples, _ = surface.shape
        part_ids = torch.arange(primitives, device=surface.device).repeat_interleave(samples)
        flat_points = surface.reshape(batch, primitives * samples, 3)
        weights = torch.sigmoid(exist_logit.squeeze(-1) / self.tau)
        weights = weights.repeat_interleave(samples, dim=1)

        return SQSurfaceSample(
            canonical_points=canonical,
            surface_points=surface,
            flat_points=flat_points,
            part_ids=part_ids,
            weights=weights,
        )
