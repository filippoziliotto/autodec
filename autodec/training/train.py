import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from autodec.training.builders import (
    build_dataloaders,
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
    build_visualizer,
    build_wandb_run,
    cfg_get,
    set_seed,
)
from autodec.training.metric_logger import EpochMetricLogger
from autodec.training.trainer import AutoDecTrainer, is_main_process
from autodec.utils.checkpoints import load_autodec_checkpoint


def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


@hydra.main(config_path="../configs", config_name="smoke", version_base=None)
def main(cfg: DictConfig):
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed + local_rank)

    model = build_model(cfg, map_location=device).to(device)
    loss_fn = build_loss(cfg).to(device)
    dataloaders, train_sampler = build_dataloaders(cfg, is_distributed=is_distributed)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(dataloaders["train"]))
    visualization_cfg = cfg_get(cfg, "visualization")
    wandb_run = build_wandb_run(cfg) if is_main_process() else None
    visualizer = build_visualizer(cfg) if is_main_process() else None

    start_epoch = 0
    best_val_loss = float("inf")
    resume_from = getattr(cfg.checkpoints, "resume_from", None)
    if resume_from is not None:
        meta = load_autodec_checkpoint(
            model,
            resume_from,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            load_optimizer=getattr(cfg.checkpoints, "keep_epoch", False),
        )
        if getattr(cfg.checkpoints, "keep_epoch", False):
            start_epoch = meta["epoch"] + 1
        best_val_loss = meta["val_loss"]

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    cfg.trainer.save_path = os.path.join(cfg.trainer.save_path, cfg.run_name)
    if is_main_process():
        os.makedirs(cfg.trainer.save_path, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.trainer.save_path, "config.yaml"))
    metric_logger = None
    if is_main_process() and cfg_get(cfg.trainer, "log_metrics_to_file", True):
        metric_logger = EpochMetricLogger(
            os.path.join(
                cfg.trainer.save_path,
                cfg_get(cfg.trainer, "metrics_log_filename", "metrics.jsonl"),
            ),
            append=getattr(cfg.checkpoints, "keep_epoch", False),
        )

    trainer = AutoDecTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        loss_fn=loss_fn,
        ctx=cfg.trainer,
        device=device,
        wandb_run=wandb_run,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        is_distributed=is_distributed,
        train_sampler=train_sampler,
        visualizer=visualizer,
        visualize_every_n_epochs=cfg_get(visualization_cfg, "every_n_epochs", 1),
        visualize_num_samples=cfg_get(visualization_cfg, "num_samples", 1),
        visualize_split=cfg_get(visualization_cfg, "split", "val"),
        log_visualizations_to_wandb=cfg_get(visualization_cfg, "log_to_wandb", True),
        metric_logger=metric_logger,
        visualize_category_balanced=cfg_get(visualization_cfg, "category_balanced", True),
        visualize_samples_per_category=cfg_get(
            visualization_cfg,
            "samples_per_category",
            cfg_get(visualization_cfg, "num_samples_per_cat", 1),
        ),
    )
    trainer.train()

    if wandb_run is not None:
        wandb_run.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
