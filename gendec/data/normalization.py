from pathlib import Path

import torch


def _stats_for(tokens, stats):
    return {
        "mean": stats["mean"].to(device=tokens.device, dtype=tokens.dtype),
        "std": stats["std"].to(device=tokens.device, dtype=tokens.dtype),
    }


def compute_normalization_stats(tokens):
    flat = tokens.reshape(-1, tokens.shape[-1]).to(torch.float32)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {"mean": mean, "std": std}


def normalize_tokens(tokens, stats):
    stats = _stats_for(tokens, stats)
    return (tokens - stats["mean"]) / stats["std"]


def unnormalize_tokens(tokens, stats):
    stats = _stats_for(tokens, stats)
    return tokens * stats["std"] + stats["mean"]


def save_normalization_stats(path, stats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)


def load_normalization_stats(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)
