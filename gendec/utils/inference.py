import math

import torch


def _exist_probability(outdict):
    if "exist" in outdict:
        return outdict["exist"].squeeze(-1)
    return torch.sigmoid(outdict["exist_logit"].squeeze(-1))


def _resample_points(points, target_count):
    if target_count is None:
        return points
    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError("target_count must be positive when provided")
    if points.shape[0] == target_count:
        return points
    if points.shape[0] > target_count:
        indices = torch.linspace(
            0,
            points.shape[0] - 1,
            steps=target_count,
            device=points.device,
        ).long()
        return points[indices]

    repeats = math.ceil(target_count / points.shape[0])
    return points.repeat((repeats, 1))[:target_count]


def prune_points_by_active_primitives(
    outdict,
    points_key,
    exist_threshold=0.5,
    target_count=None,
):
    """Return points from active primitives only.

    Without ``target_count``, this returns a list of variable-length point clouds.
    With ``target_count``, it returns a dense tensor ``[B, target_count, 3]`` by
    deterministically downsampling or repeating the pruned points.
    """

    points_by_part = outdict[points_key]
    part_ids = outdict["part_ids"].to(device=points_by_part.device)
    exist = _exist_probability(outdict).to(device=points_by_part.device)

    pruned = []
    for batch_idx in range(points_by_part.shape[0]):
        active = exist[batch_idx] > exist_threshold
        if not active.any():
            active[exist[batch_idx].argmax()] = True
        point_mask = active[part_ids]
        points = points_by_part[batch_idx, point_mask]
        pruned.append(_resample_points(points, target_count))

    if target_count is None:
        return pruned
    return torch.stack(pruned, dim=0)


def prune_decoded_points(outdict, exist_threshold=0.5, target_count=None):
    return prune_points_by_active_primitives(
        outdict,
        "decoded_points",
        exist_threshold=exist_threshold,
        target_count=target_count,
    )
