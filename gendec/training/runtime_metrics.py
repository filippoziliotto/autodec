import torch

from gendec.data.normalization import unnormalize_tokens
from gendec.losses.objectives import reconstruct_clean_tokens
from gendec.models.rotation import rot6d_to_matrix
from gendec.tokens import TOKEN_DIM, split_scaffold_tokens


def _unnormalize_pair(batch, v_hat):
    predicted = reconstruct_clean_tokens(batch, v_hat)
    mean = batch.get("token_mean")
    std = batch.get("token_std")
    if mean is None:
        mean = predicted.new_zeros(predicted.shape[-1])
    if std is None:
        std = predicted.new_ones(predicted.shape[-1])
    return unnormalize_tokens(predicted, {"mean": mean, "std": std}), unnormalize_tokens(
        batch["E0"], {"mean": mean, "std": std}
    )


def clean_token_field_mse(batch, v_hat):
    predicted_raw, target_raw = _unnormalize_pair(batch, v_hat)
    predicted_split = split_scaffold_tokens(predicted_raw)
    target_split = split_scaffold_tokens(target_raw)
    return {
        "field_mse_scale": float((predicted_split["scale"] - target_split["scale"]).pow(2).mean().detach().item()),
        "field_mse_shape": float((predicted_split["shape"] - target_split["shape"]).pow(2).mean().detach().item()),
        "field_mse_translation": float(
            (predicted_split["trans"] - target_split["trans"]).pow(2).mean().detach().item()
        ),
        "field_mse_rotation6d": float(
            (predicted_split["rot6d"] - target_split["rot6d"]).pow(2).mean().detach().item()
        ),
        "field_mse_existence": float(
            (predicted_split["exist_logit"] - target_split["exist_logit"]).pow(2).mean().detach().item()
        ),
    }


def existence_prediction_metrics(batch, v_hat):
    predicted_raw, _ = _unnormalize_pair(batch, v_hat)
    exist_logit = split_scaffold_tokens(predicted_raw)["exist_logit"]
    probs = torch.sigmoid(exist_logit)
    entropy = -(probs * probs.clamp_min(1e-8).log() + (1.0 - probs) * (1.0 - probs).clamp_min(1e-8).log())
    confident = ((probs <= 0.1) | (probs >= 0.9)).to(torch.float32)
    return {
        "exist_entropy": float(entropy.mean().detach().item()),
        "exist_confident_fraction": float(confident.mean().detach().item()),
    }


def teacher_active_count_metrics(batch, threshold=0.5):
    active = (batch["exist"].squeeze(-1) > threshold).to(torch.float32).sum(dim=-1)
    return {
        "teacher_active_primitive_count_mean": float(active.mean().detach().item()),
        "teacher_active_primitive_count_std": float(active.std(unbiased=False).detach().item()),
    }


def sample_scaffold_metrics(processed, valid_shape_range=(0.1, 2.0), orthonormal_tol=1e-3):
    tokens = processed["tokens"]
    split = split_scaffold_tokens(tokens)
    active_counts = processed["active_mask"].to(torch.float32).sum(dim=-1)

    scale_positive = (split["scale"] > 0).to(torch.float32)
    shape_valid = (
        (split["shape"] >= float(valid_shape_range[0])) & (split["shape"] <= float(valid_shape_range[1]))
    ).to(torch.float32)

    rotation = rot6d_to_matrix(split["rot6d"])
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    gram_error = torch.matmul(rotation.transpose(-1, -2), rotation) - identity
    orthonormal = (gram_error.pow(2).sum(dim=(-2, -1)).sqrt() <= orthonormal_tol).to(torch.float32)

    return {
        "active_primitive_count_mean": float(active_counts.mean().detach().item()),
        "active_primitive_count_std": float(active_counts.std(unbiased=False).detach().item()),
        "primitive_count_after_threshold_mean": float(active_counts.mean().detach().item()),
        "primitive_count_after_threshold_std": float(active_counts.std(unbiased=False).detach().item()),
        "positive_scale_fraction": float(scale_positive.mean().detach().item()),
        "valid_shape_fraction": float(shape_valid.mean().detach().item()),
        "rotation_orthonormal_fraction": float(orthonormal.mean().detach().item()),
    }


# ---------------------------------------------------------------------------
# Phase 2 metrics
# ---------------------------------------------------------------------------

def clean_joint_token_field_mse(batch, v_hat_e, explicit_dim=TOKEN_DIM):
    """Per-field MSE on the explicit branch only, reconstructing clean E tokens.

    Mirrors ``clean_token_field_mse`` but operates on the Phase 2 batch where
    ``Et`` is the full joint token and ``v_hat_e`` covers only the explicit slice.
    """
    t_tokens = batch["t"].view(-1, 1, 1)
    et_e = batch["Et"][..., :explicit_dim]
    e0_hat = et_e - t_tokens * v_hat_e

    mean = batch.get("token_mean")
    std = batch.get("token_std")
    if mean is None:
        mean = e0_hat.new_zeros(explicit_dim)
    if std is None:
        std = e0_hat.new_ones(explicit_dim)
    mean_e = mean[..., :explicit_dim]
    std_e = std[..., :explicit_dim]

    predicted_raw = unnormalize_tokens(e0_hat, {"mean": mean_e, "std": std_e})
    target_raw = unnormalize_tokens(batch["E0"][..., :explicit_dim], {"mean": mean_e, "std": std_e})

    predicted_split = split_scaffold_tokens(predicted_raw)
    target_split = split_scaffold_tokens(target_raw)
    return {
        "field_mse_scale": float((predicted_split["scale"] - target_split["scale"]).pow(2).mean().detach().item()),
        "field_mse_shape": float((predicted_split["shape"] - target_split["shape"]).pow(2).mean().detach().item()),
        "field_mse_translation": float(
            (predicted_split["trans"] - target_split["trans"]).pow(2).mean().detach().item()
        ),
        "field_mse_rotation6d": float(
            (predicted_split["rot6d"] - target_split["rot6d"]).pow(2).mean().detach().item()
        ),
        "field_mse_existence": float(
            (predicted_split["exist_logit"] - target_split["exist_logit"]).pow(2).mean().detach().item()
        ),
    }


def residual_norm_metrics(v_hat_z, batch, explicit_dim=TOKEN_DIM):
    """Statistics on the generated residual velocity and reconstructed clean Z."""
    t_tokens = batch["t"].view(-1, 1, 1)
    ez_hat_z = batch["Et"][..., explicit_dim:] - t_tokens * v_hat_z
    residual_norm = ez_hat_z.pow(2).mean(dim=-1).sqrt()  # [B, 16]
    return {
        "residual_norm_mean": float(residual_norm.mean().detach().item()),
        "residual_norm_std": float(residual_norm.std(unbiased=False).detach().item()),
    }


def sample_joint_scaffold_metrics(processed, valid_shape_range=(0.1, 2.0), orthonormal_tol=1e-3):
    """Runtime diagnostics for Phase 2 sampled joint scaffolds.

    Extends ``sample_scaffold_metrics`` with residual norm statistics.
    ``processed`` is the output of ``sample_joint_scaffolds``.
    """
    base = sample_scaffold_metrics(processed, valid_shape_range=valid_shape_range, orthonormal_tol=orthonormal_tol)
    tokens_z = processed.get("tokens_z")
    if tokens_z is not None:
        residual_norm = tokens_z.pow(2).mean(dim=-1).sqrt()  # [B, 16]
        base["residual_norm_mean"] = float(residual_norm.mean().detach().item())
        base["residual_norm_std"] = float(residual_norm.std(unbiased=False).detach().item())
    return base
