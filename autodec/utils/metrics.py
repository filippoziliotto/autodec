import torch

from autodec.losses.chamfer import weighted_chamfer_l2


def offset_ratio(surface_points, offsets, eps=1e-8):
    offset_norm = offsets.norm(dim=-1).mean()
    scaffold_norm = surface_points.norm(dim=-1).mean()
    return offset_norm / scaffold_norm.clamp_min(eps)


def active_primitive_count(exist, threshold=0.5):
    if exist.ndim == 3 and exist.shape[-1] == 1:
        exist = exist.squeeze(-1)
    return (exist > threshold).to(torch.float32).sum(dim=-1).mean()


def active_decoded_point_count(weights, threshold=0.5):
    if weights.ndim == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    return (weights > threshold).to(torch.float32).sum(dim=-1).mean()


def primitive_mass_entropy(assign_matrix, eps=1e-8):
    mass = assign_matrix.mean(dim=1)
    probs = mass / mass.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
    return entropy.mean()


def scaffold_vs_decoded_chamfer(
    surface_points,
    decoded_points,
    target_points,
    weights,
    chamfer_fn=weighted_chamfer_l2,
):
    scaffold = chamfer_fn(surface_points, target_points, weights)
    decoded = chamfer_fn(decoded_points, target_points, weights)
    return {
        "scaffold_chamfer": scaffold.detach().item(),
        "decoded_chamfer": decoded.detach().item(),
    }
