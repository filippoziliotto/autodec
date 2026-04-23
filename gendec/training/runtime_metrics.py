import torch

from gendec.data.normalization import unnormalize_tokens
from gendec.losses.objectives import reconstruct_clean_tokens
from gendec.models.rotation import rot6d_to_matrix
from gendec.tokens import split_scaffold_tokens


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
