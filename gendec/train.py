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

from gendec.config import fallback_cli_config
from gendec.training.builders import (
    build_scheduler,
    build_train_val_dataloaders,
    build_loss,
    build_model,
    build_optimizer,
    build_wandb_run,
    cfg_get,
    set_seed,
)
from gendec.training.trainer import Phase1Trainer


def run_train(cfg):
    set_seed(cfg_get(cfg, "seed", 0))
    datasets, dataloaders = build_train_val_dataloaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(dataloaders["train"]))
    wandb_run = build_wandb_run(cfg)
    try:
        trainer = Phase1Trainer(
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
    result = run_train(cfg)
    print(result)


if hydra is None:
    if __name__ == "__main__":
        _main(fallback_cli_config("train.yaml"))
else:
    main = hydra.main(config_path="configs", config_name="train", version_base=None)(_main)
    if __name__ == "__main__":
        main()
