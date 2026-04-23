import random
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from gendec.config import cfg_get
from gendec.data.dataset import ScaffoldTokenDataset
from gendec.losses.flow_matching import FlowMatchingLoss
from gendec.models.set_transformer_flow import SetTransformerFlowModel
from gendec.training.schedulers import build_cosine_warmup_scheduler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _import_wandb():
    import wandb

    return wandb


def build_wandb_run(cfg):
    if not cfg_get(cfg, "use_wandb", False):
        return None
    wandb = _import_wandb()
    wandb_cfg = cfg_get(cfg, "wandb")
    api_key_env = cfg_get(wandb_cfg, "api_key_env", "WANDB_API_KEY")
    if api_key_env and api_key_env != "WANDB_API_KEY" and os.environ.get(api_key_env):
        os.environ.setdefault("WANDB_API_KEY", os.environ[api_key_env])
    kwargs = {
        "project": cfg_get(wandb_cfg, "project", "gendec"),
        "name": cfg_get(cfg, "run_name", None),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return wandb.init(**kwargs)


def _training_cfg(cfg):
    return cfg_get(cfg, "training", cfg_get(cfg, "trainer"))


def _sampling_cfg(cfg):
    return cfg_get(cfg, "sampling", cfg_get(cfg, "sampler"))


def build_dataset(cfg, split=None):
    dataset_cfg = cfg_get(cfg, "dataset")
    return ScaffoldTokenDataset(
        root=cfg_get(dataset_cfg, "root"),
        split=cfg_get(dataset_cfg, "split", None) if split is None else split,
        categories=cfg_get(dataset_cfg, "categories", None),
    )


def build_dataloader(cfg, split=None, batch_size=None, shuffle=None):
    dataset = build_dataset(cfg, split=split)
    trainer_cfg = _training_cfg(cfg)
    resolved_split = cfg_get(cfg_get(cfg, "dataset"), "split", None) if split is None else split
    if batch_size is None:
        batch_size = cfg_get(trainer_cfg, "batch_size", 1)
    if shuffle is None:
        shuffle = resolved_split in {None, "train"}
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg_get(trainer_cfg, "num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, dataloader


def build_train_val_dataloaders(cfg):
    dataset_cfg = cfg_get(cfg, "dataset")
    train_split = cfg_get(dataset_cfg, "split", "train")
    val_split = cfg_get(dataset_cfg, "val_split", "val")

    train_dataset, train_loader = build_dataloader(cfg, split=train_split, shuffle=True)
    try:
        val_dataset, val_loader = build_dataloader(cfg, split=val_split, shuffle=False)
    except FileNotFoundError:
        val_dataset, val_loader = None, None
    return {"train": train_dataset, "val": val_dataset}, {"train": train_loader, "val": val_loader}


def build_model(cfg):
    model_cfg = cfg_get(cfg, "model")
    return SetTransformerFlowModel(
        token_dim=cfg_get(model_cfg, "token_dim", 15),
        hidden_dim=cfg_get(model_cfg, "hidden_dim", 256),
        n_blocks=cfg_get(model_cfg, "n_blocks", 6),
        n_heads=cfg_get(model_cfg, "n_heads", 8),
        dropout=cfg_get(model_cfg, "dropout", 0.0),
    )


def build_loss(cfg):
    loss_cfg = cfg_get(cfg, "loss")
    return FlowMatchingLoss(
        lambda_flow=cfg_get(loss_cfg, "lambda_flow", 1.0),
        lambda_exist=cfg_get(loss_cfg, "lambda_exist", 0.05),
        exist_channel=cfg_get(loss_cfg, "exist_channel", -1),
    )


def build_optimizer(cfg, model):
    opt_cfg = cfg_get(cfg, "optimizer")
    name = str(cfg_get(opt_cfg, "name", "AdamW")).lower()
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer {name!r}; only AdamW is implemented for gendec Phase 1")
    return AdamW(
        model.parameters(),
        lr=cfg_get(opt_cfg, "lr", 1e-4),
        weight_decay=cfg_get(opt_cfg, "weight_decay", 1e-4),
        betas=tuple(cfg_get(opt_cfg, "betas", (0.9, 0.999))),
        eps=cfg_get(opt_cfg, "eps", 1e-8),
    )


def build_scheduler(cfg, optimizer, steps_per_epoch):
    scheduler_cfg = cfg_get(cfg, "scheduler")
    if scheduler_cfg is None:
        return None
    name = cfg_get(scheduler_cfg, "name")
    if name is None or str(name).lower() in {"", "none"}:
        return None
    if str(name).lower() != "cosine":
        raise ValueError(f"Unsupported scheduler {name!r}; only cosine is implemented for gendec Phase 1")
    training_cfg = _training_cfg(cfg)
    total_steps = max(int(cfg_get(training_cfg, "num_epochs", 1)) * int(steps_per_epoch), 1)
    return build_cosine_warmup_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=cfg_get(scheduler_cfg, "warmup_steps", 0),
        min_lr=cfg_get(scheduler_cfg, "min_lr", 1e-5),
    )
