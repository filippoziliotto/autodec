from pathlib import Path

import torch


def strip_module_prefix(state_dict):
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def save_phase1_checkpoint(model, optimizer, scheduler, epoch, loss, path, ema_model=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "ema_model_state_dict": None if ema_model is None else ema_model.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )
    return path


def load_phase1_checkpoint(
    model,
    path,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    load_optimizer=False,
    use_ema=True,
):
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    state_dict_key = "ema_model_state_dict" if use_ema and checkpoint.get("ema_model_state_dict") is not None else "model_state_dict"
    state_dict = strip_module_prefix(checkpoint[state_dict_key])
    model.load_state_dict(state_dict)
    if load_optimizer and optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if load_optimizer and scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return {
        "epoch": checkpoint.get("epoch", -1),
        "loss": checkpoint.get("loss", float("inf")),
    }
