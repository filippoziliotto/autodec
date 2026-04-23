import torch

from gendec.data.normalization import unnormalize_tokens
from gendec.models.rotation import rot6d_to_matrix
from gendec.tokens import PRIMITIVE_COUNT, TOKEN_DIM, split_scaffold_tokens


def euler_sample(model, num_samples, token_dim=TOKEN_DIM, num_steps=50, device="cpu"):
    model.eval()
    tokens = torch.randn(num_samples, PRIMITIVE_COUNT, token_dim, device=device)
    time_grid = torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)
    with torch.no_grad():
        for idx in range(num_steps):
            t_cur = time_grid[idx].expand(num_samples)
            dt = time_grid[idx] - time_grid[idx + 1]
            velocity = model(tokens, t_cur)
            tokens = tokens - velocity * dt.view(1, 1, 1)
    return tokens


def postprocess_tokens(tokens, stats, exist_threshold=0.5):
    raw = unnormalize_tokens(tokens, stats)
    split = split_scaffold_tokens(raw)
    split["scale"] = split["scale"].clamp_min(1e-3)
    split["shape"] = split["shape"].clamp(0.1, 2.0)
    split["rotate"] = rot6d_to_matrix(split["rot6d"])
    split["exist"] = torch.sigmoid(split["exist_logit"])
    split["active_mask"] = split["exist"].squeeze(-1) > exist_threshold
    split["tokens"] = raw
    return split


def _signed_power(value, exponent, eps=1e-6):
    return torch.sign(value) * value.abs().clamp_min(eps).pow(exponent)


def render_scaffold_preview(processed, points_per_primitive=64):
    device = processed["tokens"].device
    batch, n_primitives = processed["scale"].shape[:2]
    n_eta = max(2, int(points_per_primitive**0.5))
    n_omega = max(2, points_per_primitive // n_eta)
    etas = torch.linspace(-torch.pi / 2, torch.pi / 2, steps=n_eta, device=device)
    omegas = torch.linspace(-torch.pi, torch.pi, steps=n_omega, device=device)
    eta_grid, omega_grid = torch.meshgrid(etas, omegas, indexing="ij")
    eta_grid = eta_grid.reshape(1, 1, -1)
    omega_grid = omega_grid.reshape(1, 1, -1)

    scale = processed["scale"]
    shape = processed["shape"]
    e1 = shape[..., 0:1]
    e2 = shape[..., 1:2]
    x = scale[..., 0:1] * _signed_power(torch.cos(eta_grid), e1) * _signed_power(torch.cos(omega_grid), e2)
    y = scale[..., 1:2] * _signed_power(torch.cos(eta_grid), e1) * _signed_power(torch.sin(omega_grid), e2)
    z = scale[..., 2:3] * _signed_power(torch.sin(eta_grid), e1)
    canonical = torch.stack([x, y, z], dim=-1)
    world = torch.matmul(processed["rotate"].unsqueeze(2), canonical.unsqueeze(-1)).squeeze(-1)
    world = world + processed["trans"].unsqueeze(2)

    preview_points = []
    for batch_idx in range(batch):
        active = processed["active_mask"][batch_idx]
        if active.any():
            preview_points.append(world[batch_idx, active].reshape(-1, 3))
        else:
            preview_points.append(world.new_zeros(0, 3))

    max_points = max(points.shape[0] for points in preview_points)
    padded = world.new_zeros(batch, max_points, 3)
    for batch_idx, points in enumerate(preview_points):
        if points.numel() > 0:
            padded[batch_idx, : points.shape[0]] = points
    return padded


def sample_scaffolds(
    model,
    stats,
    num_samples,
    token_dim=TOKEN_DIM,
    num_steps=50,
    exist_threshold=0.5,
    device="cpu",
):
    tokens = euler_sample(model, num_samples=num_samples, token_dim=token_dim, num_steps=num_steps, device=device)
    processed = postprocess_tokens(tokens, stats=stats, exist_threshold=exist_threshold)
    preview_points = render_scaffold_preview(processed)
    processed["preview_points"] = preview_points
    return processed
