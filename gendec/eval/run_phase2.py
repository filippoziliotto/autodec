import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

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
from gendec.eval.evaluator import Phase2Evaluator
from gendec.training.builders import build_phase2_dataset, build_phase2_loss, build_phase2_model, set_seed
from gendec.training.checkpoints import load_phase1_checkpoint


def run_eval_phase2(cfg):
    set_seed(cfg_get(cfg, "seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_phase2_model(cfg).to(device)
    checkpoints_cfg = cfg_get(cfg, "checkpoints", cfg_get(cfg, "training"))
    resume_from = cfg_get(checkpoints_cfg, "resume_from", cfg_get(checkpoints_cfg, "best_checkpoint_path"))
    load_phase1_checkpoint(model, resume_from, map_location=device)
    loss_fn = build_phase2_loss(cfg)
    dataset = build_phase2_dataset(cfg)
    evaluator = Phase2Evaluator(cfg=cfg, model=model, loss_fn=loss_fn, dataset=dataset, device=device)
    return evaluator.evaluate()


def _main(cfg: DictConfig):
    result = run_eval_phase2(cfg)
    print(result["metrics"])


if hydra is None:
    if __name__ == "__main__":
        _main(fallback_cli_config("eval_phase2.yaml"))
else:
    main = hydra.main(config_path="../configs", config_name="eval_phase2", version_base=None)(_main)
    if __name__ == "__main__":
        explicit_config = explicit_config_argument("eval_phase2.yaml")
        if explicit_config is not None:
            _main(load_yaml_config(explicit_config))
        else:
            main()
