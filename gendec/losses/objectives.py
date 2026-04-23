import torch
import torch.nn.functional as F


def per_sample_flow_mse(v_hat, velocity_target):
    return (v_hat - velocity_target).pow(2).reshape(v_hat.shape[0], -1).mean(dim=1)


def reconstruct_clean_tokens(batch, v_hat):
    t_tokens = batch["t"].view(-1, 1, 1)
    return batch["Et"] - t_tokens * v_hat


def unnormalize_exist_logits(tokens, mean, std, exist_channel):
    return tokens[..., exist_channel] * std[exist_channel] + mean[exist_channel]


def per_sample_exist_bce(batch, v_hat, exist_channel):
    e0_hat = reconstruct_clean_tokens(batch, v_hat)
    mean = batch.get("token_mean")
    std = batch.get("token_std")
    if mean is None:
        mean = e0_hat.new_zeros(e0_hat.shape[-1])
    if std is None:
        std = e0_hat.new_ones(e0_hat.shape[-1])
    logits = unnormalize_exist_logits(e0_hat, mean, std, exist_channel)
    target = (batch["exist"].squeeze(-1) > 0.5).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return bce.mean(dim=1)
