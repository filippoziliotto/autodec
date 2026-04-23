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

from gendec.config import explicit_config_argument, fallback_cli_config, load_yaml_config
from gendec.data.layout import normalization_stats_path
from gendec.data.normalization import load_normalization_stats
from gendec.sampling import sample_scaffolds
from gendec.training.builders import build_model, cfg_get
from gendec.training.checkpoints import load_phase1_checkpoint


def run_sample(cfg):
    dataset_cfg = cfg_get(cfg, "dataset")
    sampler_cfg = cfg_get(cfg, "sampling", cfg_get(cfg, "sampler"))
    trainer_cfg = cfg_get(cfg, "training", cfg_get(cfg, "trainer", cfg_get(cfg, "checkpoints")))
    root = Path(cfg_get(dataset_cfg, "root"))
    stats = load_normalization_stats(normalization_stats_path(root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint_path = cfg_get(
        trainer_cfg,
        "best_checkpoint_path",
        cfg_get(trainer_cfg, "checkpoint_path", cfg_get(trainer_cfg, "resume_from")),
    )
    load_phase1_checkpoint(model, checkpoint_path, map_location=device)
    result = sample_scaffolds(
        model=model,
        stats=stats,
        num_samples=cfg_get(sampler_cfg, "num_samples", 1),
        token_dim=cfg_get(cfg_get(cfg, "model"), "token_dim", 15),
        num_steps=cfg_get(sampler_cfg, "eval_steps", cfg_get(sampler_cfg, "num_steps", 50)),
        exist_threshold=cfg_get(sampler_cfg, "exist_threshold", 0.5),
        device=device,
    )
    output_dir = Path(cfg_get(sampler_cfg, "output_dir", root / "samples"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "samples.pt"
    torch.save(
        {
            "tokens": result["tokens"].cpu(),
            "exist": result["exist"].cpu(),
            "active_mask": result["active_mask"].cpu(),
            "preview_points": result["preview_points"].cpu(),
        },
        output_path,
    )
    result["output_path"] = output_path
    return result


def _main(cfg: DictConfig):
    result = run_sample(cfg)
    print(result["output_path"])


if hydra is None:
    if __name__ == "__main__":
        _main(fallback_cli_config("sample.yaml"))
else:
    main = hydra.main(config_path="configs", config_name="sample", version_base=None)(_main)
    if __name__ == "__main__":
        explicit_config = explicit_config_argument("sample.yaml")
        if explicit_config is not None:
            _main(load_yaml_config(explicit_config))
        else:
            main()
