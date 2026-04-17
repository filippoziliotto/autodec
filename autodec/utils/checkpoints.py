from pathlib import Path

import torch


def strip_module_prefix(state_dict):
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _load_checkpoint(path, map_location="cpu"):
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def _encoder_target(model_or_encoder):
    return getattr(model_or_encoder, "encoder", model_or_encoder)


def load_superdec_encoder_checkpoint(
    model_or_encoder,
    path,
    map_location="cpu",
    strict=True,
):
    checkpoint = _load_checkpoint(path, map_location=map_location)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    target = _encoder_target(model_or_encoder)
    return target.load_state_dict(state_dict, strict=strict)


def load_autodec_checkpoint(
    model,
    path,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    load_optimizer=True,
):
    checkpoint = _load_checkpoint(path, map_location=map_location)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict)

    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if load_optimizer and scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", -1),
        "val_loss": checkpoint.get("val_loss", float("inf")),
    }


def save_autodec_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)
    return path
