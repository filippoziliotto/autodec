import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import hydra
except ModuleNotFoundError:
    hydra = None

try:
    from omegaconf import DictConfig
except ModuleNotFoundError:
    DictConfig = object

import torch

from gendec.config import explicit_config_argument, fallback_cli_config, cfg_get, load_yaml_config
from gendec.training.builders import (
    build_optimizer,
    build_phase2_loss,
    build_phase2_model,
    build_phase2_train_val_dataloaders,
    build_scheduler,
    build_wandb_run,
    set_seed,
)
from gendec.training.checkpoints import load_phase1_checkpoint
from gendec.training.trainer import Phase2Trainer


def run_train_phase2(cfg):
    set_seed(cfg_get(cfg, "seed", 0))
    datasets, dataloaders = build_phase2_train_val_dataloaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_phase2_model(cfg).to(device)

    checkpoint_cfg = cfg_get(cfg, "checkpoints", None)
    resume_from = cfg_get(checkpoint_cfg, "resume_from", None)
    if resume_from is not None:
        load_phase1_checkpoint(model, resume_from, map_location=device, load_optimizer=False)

    loss_fn = build_phase2_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(dataloaders["train"]))
    wandb_run = build_wandb_run(cfg)
    try:
        trainer = Phase2Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders["val"],
            scheduler=scheduler,
            stats=datasets["train"].stats,
            cfg=cfg,
            device=device,
            wandb_run=wandb_run,
        )
        result = trainer.train()
        result["num_train_examples"] = len(datasets["train"])
        result["num_val_examples"] = 0 if datasets["val"] is None else len(datasets["val"])
        return result
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _main(cfg: DictConfig):
    result = run_train_phase2(cfg)
    print(result)


if hydra is None:
    if __name__ == "__main__":
        _main(fallback_cli_config("train_phase2.yaml"))
else:
    main = hydra.main(config_path="configs", config_name="train_phase2", version_base=None)(_main)
    if __name__ == "__main__":
        explicit_config = explicit_config_argument("train_phase2.yaml")
        if explicit_config is not None:
            _main(load_yaml_config(explicit_config))
        else:
            main()
