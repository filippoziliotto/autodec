import os
from pathlib import Path

import torch

from autodec.eval.evaluator import AutoDecEvaluator
from autodec.training.builders import (
    build_loss,
    build_model,
    build_wandb_run,
    cfg_get,
    set_seed,
)
from autodec.utils.checkpoints import load_autodec_checkpoint
from autodec.visualizations import AutoDecEpochVisualizer

try:
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:
    DictConfig = object
    OmegaConf = None

try:
    import hydra
except ModuleNotFoundError:
    hydra = None


def _build_dataset(cfg):
    if cfg_get(cfg, "dataset", "shapenet") != "shapenet":
        raise ValueError("AutoDec test evaluation currently supports dataset=shapenet")
    from superdec.data.dataloader import ShapeNet

    eval_cfg = cfg_get(cfg, "eval")
    return ShapeNet(split=cfg_get(eval_cfg, "split", "test"), cfg=cfg)


def _build_eval_visualizer(cfg):
    vis_cfg = cfg_get(cfg, "visualization")
    if not cfg_get(vis_cfg, "enabled", True):
        return None
    return AutoDecEpochVisualizer(
        root_dir=cfg_get(
            vis_cfg,
            "root_dir",
            cfg_get(vis_cfg, "output_dir", cfg_get(cfg_get(cfg, "eval"), "output_dir", "data/eval")),
        ),
        run_name=cfg_get(vis_cfg, "run_name", cfg_get(cfg, "run_name", "autodec_test_eval")),
        mesh_resolution=cfg_get(vis_cfg, "mesh_resolution", 24),
        exist_threshold=cfg_get(vis_cfg, "exist_threshold", 0.5),
        max_points=cfg_get(vis_cfg, "max_points", 4096),
    )


def maybe_enable_lm_optimization(model, cfg, device):
    eval_cfg = cfg_get(cfg, "eval")
    if not cfg_get(eval_cfg, "use_lm_optimization", False):
        return False
    if device.type != "cuda":
        raise RuntimeError(
            "eval.use_lm_optimization requires CUDA because SuperDec LMOptimizer "
            "uses CUDA-only operations."
        )
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "enable_lm_optimization"):
        raise TypeError("LM optimization requires a model with AutoDecEncoder")
    encoder.enable_lm_optimization()
    return True


def _main(cfg: DictConfig):
    if OmegaConf is None:
        raise ModuleNotFoundError("omegaconf is required to run AutoDec evaluation")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    checkpoint = cfg_get(cfg_get(cfg, "checkpoints"), "resume_from")
    if checkpoint is None:
        raise ValueError("Evaluation requires checkpoints.resume_from")

    model = build_model(cfg, map_location=device).to(device)
    load_autodec_checkpoint(
        model,
        checkpoint,
        map_location=device,
        load_optimizer=False,
    )
    maybe_enable_lm_optimization(model, cfg, device)
    loss_fn = build_loss(cfg).to(device)
    dataset = _build_dataset(cfg)
    visualizer = _build_eval_visualizer(cfg)
    wandb_run = build_wandb_run(cfg)

    output_dir = Path(cfg.eval.output_dir) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    evaluator = AutoDecEvaluator(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        dataset=dataset,
        visualizer=visualizer,
        device=device,
        wandb_run=wandb_run,
    )
    result = evaluator.evaluate()

    if wandb_run is not None:
        wandb_run.finish()

    print(f"Wrote AutoDec test evaluation to {os.fspath(output_dir)}")
    print(OmegaConf.to_yaml(OmegaConf.create(result["metrics"])))


if hydra is None:
    def main(*args, **kwargs):
        raise ModuleNotFoundError("hydra is required to run AutoDec evaluation")
else:
    main = hydra.main(config_path="../configs", config_name="eval_test", version_base=None)(_main)


if __name__ == "__main__":
    main()
