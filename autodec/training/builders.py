import os
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from autodec.autodec import AutoDec
from autodec.losses import AutoDecLoss
from autodec.utils.checkpoints import load_superdec_encoder_checkpoint
from autodec.visualizations import AutoDecEpochVisualizer


def cfg_get(cfg, name, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _phase_number(phase):
    if isinstance(phase, str):
        text = phase.lower().replace("_", "").replace("-", "")
        if text.startswith("phase"):
            text = text[5:]
        if text == "decoderwarmup":
            return 1
        if text == "joint":
            return 2
        return int(text)
    return int(phase)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
        "project": cfg_get(wandb_cfg, "project", "autodec"),
        "name": cfg_get(cfg, "run_name", None),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return wandb.init(**kwargs)


def build_visualizer(cfg):
    vis_cfg = cfg_get(cfg, "visualization")
    if not cfg_get(vis_cfg, "enabled", False):
        return None
    return AutoDecEpochVisualizer(
        root_dir=cfg_get(vis_cfg, "root_dir", "data/viz"),
        run_name=cfg_get(vis_cfg, "run_name", cfg_get(cfg, "run_name", "autodec")),
        mesh_resolution=cfg_get(vis_cfg, "mesh_resolution", 24),
        exist_threshold=cfg_get(vis_cfg, "exist_threshold", 0.5),
        max_points=cfg_get(vis_cfg, "max_points", 4096),
    )


def build_model(cfg, map_location="cpu"):
    model = AutoDec(cfg_get(cfg, "autodec"))
    checkpoint_cfg = cfg_get(cfg, "checkpoints")
    encoder_from = cfg_get(checkpoint_cfg, "encoder_from")
    if encoder_from is not None:
        load_superdec_encoder_checkpoint(
            model,
            encoder_from,
            map_location=map_location,
            strict=cfg_get(checkpoint_cfg, "strict_encoder", True),
        )
    return model


def build_loss(cfg):
    loss_cfg = cfg_get(cfg, "loss")
    return AutoDecLoss(
        phase=cfg_get(loss_cfg, "phase", 1),
        lambda_sq=cfg_get(loss_cfg, "lambda_sq", cfg_get(loss_cfg, "w_sq", 1.0)),
        lambda_par=cfg_get(loss_cfg, "lambda_par", cfg_get(loss_cfg, "w_par", 0.06)),
        lambda_exist=cfg_get(loss_cfg, "lambda_exist", cfg_get(loss_cfg, "w_exist", 0.01)),
        lambda_cons=cfg_get(loss_cfg, "lambda_cons", cfg_get(loss_cfg, "w_cons", 0.0)),
        n_sq_samples=cfg_get(loss_cfg, "n_sq_samples", cfg_get(loss_cfg, "n_samples", 256)),
        sq_tau=cfg_get(loss_cfg, "sq_tau", 1.0),
        exist_point_threshold=cfg_get(loss_cfg, "exist_point_threshold", 24.0),
        active_exist_threshold=cfg_get(loss_cfg, "active_exist_threshold", 0.5),
        chamfer_eps=cfg_get(loss_cfg, "chamfer_eps", 1e-6),
        min_backward_weight=cfg_get(loss_cfg, "min_backward_weight", 1e-3),
    )


def _trainable(params):
    return [param for param in params if param.requires_grad]


def build_optimizer(cfg, model):
    opt_cfg = cfg_get(cfg, "optimizer")
    loss_cfg = cfg_get(cfg, "loss")
    phase = _phase_number(cfg_get(opt_cfg, "phase", cfg_get(loss_cfg, "phase", 1)))
    betas = tuple(cfg_get(opt_cfg, "betas", (0.9, 0.999)))
    weight_decay = cfg_get(opt_cfg, "weight_decay", 0.0)
    base_lr = cfg_get(opt_cfg, "lr", 1e-4)
    decoder_lr = cfg_get(opt_cfg, "decoder_lr", base_lr)
    residual_lr = cfg_get(opt_cfg, "residual_lr", decoder_lr)
    encoder_lr = cfg_get(opt_cfg, "encoder_lr", base_lr)

    if phase < 2:
        model.freeze_encoder_backbone()
        return Adam(
            _trainable(model.phase1_parameters()),
            lr=decoder_lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    model.unfreeze_encoder()
    groups = [
        {"params": _trainable(model.encoder_backbone_parameters()), "lr": encoder_lr},
        {"params": _trainable(model.residual_parameters()), "lr": residual_lr},
        {"params": _trainable(model.decoder_parameters()), "lr": decoder_lr},
    ]
    groups = [group for group in groups if group["params"]]
    return Adam(groups, betas=betas, weight_decay=weight_decay)


def build_scheduler(cfg, optimizer, steps_per_epoch):
    opt_cfg = cfg_get(cfg, "optimizer")
    if not cfg_get(opt_cfg, "enable_scheduler", False):
        return None
    import hydra

    scheduler_cfg = cfg_get(cfg, "scheduler")
    scheduler_cfg.steps_per_epoch = steps_per_epoch
    return hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)


def limit_dataset(dataset, max_items=None, seed=0):
    """Return a deterministic random subset when max_items is smaller than dataset."""

    if max_items is None:
        return dataset
    max_items = int(max_items)
    if max_items <= 0:
        raise ValueError("Dataset size limit must be positive when provided")
    if max_items >= len(dataset):
        return dataset

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:max_items].tolist()
    return Subset(dataset, indices)


def _limit_shapenet_splits(cfg, train_ds, val_ds):
    shapenet_cfg = cfg_get(cfg, "shapenet")
    seed = cfg_get(shapenet_cfg, "subset_seed", cfg_get(cfg, "seed", 0))
    train_limit = cfg_get(shapenet_cfg, "max_train_items")
    val_limit = cfg_get(shapenet_cfg, "max_val_items")
    return (
        limit_dataset(train_ds, max_items=train_limit, seed=seed),
        limit_dataset(val_ds, max_items=val_limit, seed=int(seed) + 1),
    )


def build_dataloaders(cfg, is_distributed=False):
    from superdec.data.dataloader import ABO, ASE_Object, ShapeNet

    dataset = cfg_get(cfg, "dataset")
    if dataset == "shapenet":
        train_ds = ShapeNet(split="train", cfg=cfg)
        val_ds = ShapeNet(split="val", cfg=cfg)
        train_ds, val_ds = _limit_shapenet_splits(cfg, train_ds, val_ds)
    elif dataset == "abo":
        train_ds = ABO(split="train", cfg=cfg)
        val_ds = ABO(split="val", cfg=cfg)
    elif dataset == "ase_object":
        train_ds = ASE_Object(split="train", cfg=cfg)
        val_ds = ASE_Object(split="val", cfg=cfg)
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    trainer_cfg = cfg_get(cfg, "trainer")
    if is_distributed:
        train_sampler = DistributedSampler(train_ds)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_get(trainer_cfg, "batch_size", 1),
        shuffle=shuffle,
        num_workers=cfg_get(trainer_cfg, "num_workers", 0),
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_get(trainer_cfg, "batch_size", 1),
        shuffle=False,
        num_workers=cfg_get(trainer_cfg, "num_workers", 0),
        pin_memory=True,
    )
    return {"train": train_loader, "val": val_loader}, train_sampler
