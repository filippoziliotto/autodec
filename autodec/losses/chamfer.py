import torch


def _check_chamfer_inputs(pred, target, weights):
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("pred must have shape [B, M, 3]")
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("target must have shape [B, N, 3]")
    if weights.ndim == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.ndim != 2:
        raise ValueError("weights must have shape [B, M] or [B, M, 1]")
    if pred.shape[:2] != weights.shape:
        raise ValueError("weights must match pred batch and point dimensions")
    if pred.shape[0] != target.shape[0]:
        raise ValueError("pred and target must have the same batch size")
    return weights


def weighted_chamfer_l2(
    pred,
    target,
    weights,
    eps=1e-6,
    min_backward_weight=1e-3,
    return_components=False,
):
    """Weighted bidirectional Chamfer-L2 for fixed-size decoded point clouds.

    The forward term weights predicted points by existence. The backward term
    divides distances by the prediction weights before nearest-neighbor
    selection, discouraging coverage through inactive scaffold points.
    """

    weights = _check_chamfer_inputs(pred, target, weights).to(
        device=pred.device,
        dtype=pred.dtype,
    )
    target = target.to(device=pred.device, dtype=pred.dtype)

    distances = torch.cdist(pred, target, p=2).pow(2)

    forward_weights = weights.clamp_min(eps)
    forward = distances.min(dim=2).values
    forward = (forward * forward_weights).sum(dim=1) / forward_weights.sum(dim=1)

    backward_weights = weights.clamp_min(min_backward_weight)
    weighted_distances = distances / backward_weights.unsqueeze(-1)
    backward = weighted_distances.min(dim=1).values.mean(dim=1)

    components = {
        "forward": forward.mean(),
        "backward": backward.mean(),
    }
    loss = components["forward"] + components["backward"]
    if return_components:
        return loss, components
    return loss
