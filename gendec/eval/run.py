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

from gendec.config import fallback_cli_config, cfg_get
from gendec.eval.evaluator import Phase1Evaluator
from gendec.training.builders import build_dataset, build_loss, build_model, set_seed
from gendec.training.checkpoints import load_phase1_checkpoint


def run_eval(cfg):
    set_seed(cfg_get(cfg, "seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoints_cfg = cfg_get(cfg, "checkpoints", cfg_get(cfg, "training"))
    load_phase1_checkpoint(model, cfg_get(checkpoints_cfg, "resume_from", cfg_get(checkpoints_cfg, "best_checkpoint_path")), map_location=device)
    loss_fn = build_loss(cfg)
    dataset = build_dataset(cfg)
    evaluator = Phase1Evaluator(cfg=cfg, model=model, loss_fn=loss_fn, dataset=dataset, device=device)
    return evaluator.evaluate()


def _main(cfg: DictConfig):
    result = run_eval(cfg)
    print(result["metrics"])


if hydra is None:
    if __name__ == "__main__":
        _main(fallback_cli_config("eval.yaml"))
else:
    main = hydra.main(config_path="../configs", config_name="eval", version_base=None)(_main)
    if __name__ == "__main__":
        main()
