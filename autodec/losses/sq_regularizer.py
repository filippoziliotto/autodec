import torch
import torch.nn as nn
import torch.nn.functional as F

from autodec.sampling.sq_surface import SQSurfaceSampler


def _batch_points(batch, reference):
    points = batch["points"]
    return points.to(device=reference.device, dtype=reference.dtype)


def _exist_logit(outdict):
    if "exist_logit" in outdict:
        return outdict["exist_logit"]
    exist = outdict["exist"].clamp(1e-6, 1 - 1e-6)
    return torch.logit(exist)


def assignment_parsimony_loss(assign_matrix, stabilizer=0.01):
    """SuperDec-style 0.5-norm sparsity surrogate over primitive mass."""

    mass = assign_matrix.mean(dim=1)
    return (mass + stabilizer).sqrt().mean(dim=1).pow(2).mean()


def existence_loss(
    assign_matrix,
    exist=None,
    exist_logit=None,
    point_threshold=24.0,
):
    """Binary existence supervision derived from assignment mass."""

    if exist_logit is None and exist is None:
        raise ValueError("existence_loss requires exist or exist_logit")

    target = (assign_matrix.sum(dim=1) > point_threshold).to(assign_matrix.dtype)
    if exist_logit is not None:
        pred = exist_logit.squeeze(-1).to(device=assign_matrix.device, dtype=assign_matrix.dtype)
        return F.binary_cross_entropy_with_logits(pred, target)

    pred = exist.squeeze(-1).to(device=assign_matrix.device, dtype=assign_matrix.dtype)
    return F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target)


class SQRegularizer(nn.Module):
    """Device-safe sampled superquadric Chamfer-L2 regularizer."""

    def __init__(self, n_samples=256, tau=1.0, angle_sampler=None, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.surface_sampler = SQSurfaceSampler(
            n_samples=n_samples,
            tau=tau,
            angle_sampler=angle_sampler,
        )

    def _sampler_outdict(self, outdict):
        if "exist_logit" in outdict:
            return outdict
        sampler_outdict = dict(outdict)
        sampler_outdict["exist_logit"] = _exist_logit(outdict)
        return sampler_outdict

    def forward(self, batch, outdict, return_components=False):
        points = _batch_points(batch, outdict["scale"])
        assign_matrix = outdict["assign_matrix"].to(
            device=points.device,
            dtype=points.dtype,
        )
        exist = outdict.get("exist")
        if exist is None:
            exist = torch.sigmoid(_exist_logit(outdict))
        exist = exist.to(device=points.device, dtype=points.dtype).squeeze(-1)

        sample = self.surface_sampler(self._sampler_outdict(outdict))
        batch_size, primitives, samples, _ = sample.surface_points.shape
        flat_surface = sample.surface_points.reshape(batch_size, primitives * samples, 3)

        distances = torch.cdist(flat_surface, points, p=2).pow(2)
        distances = distances.view(batch_size, primitives, samples, points.shape[1])

        point_to_sq = distances.min(dim=2).values.transpose(1, 2)
        point_to_sq = (point_to_sq * assign_matrix).sum(dim=-1).mean(dim=-1)
        point_to_sq = point_to_sq.mean()

        sq_to_point = distances.min(dim=3).values.mean(dim=-1)
        sq_to_point = (sq_to_point * exist).sum(dim=-1)
        sq_to_point = sq_to_point / exist.sum(dim=-1).clamp_min(self.eps)
        sq_to_point = sq_to_point.mean()

        components = {
            "point_to_sq": point_to_sq,
            "sq_to_point": sq_to_point,
        }
        loss = point_to_sq + sq_to_point
        if return_components:
            return loss, components
        return loss
