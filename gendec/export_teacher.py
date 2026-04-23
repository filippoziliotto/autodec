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

from gendec.config import explicit_config_argument, fallback_cli_config, cfg_get, load_yaml_config
from gendec.data.build_teacher_dataset import export_teacher_dataset
from gendec.data.toy_builder import write_toy_teacher_dataset_splits


def run_export(cfg):
    export_cfg = cfg_get(cfg, "export")
    if cfg_get(export_cfg, "mode", "toy") == "toy":
        return write_toy_teacher_dataset_splits(
            root=cfg_get(export_cfg, "output_root"),
            split=cfg_get(export_cfg, "split", "train"),
            splits=cfg_get(export_cfg, "splits"),
            num_examples=cfg_get(export_cfg, "num_examples", 8),
            num_points=cfg_get(export_cfg, "num_points", 4096),
        )
    return export_teacher_dataset(cfg)


def _main(cfg: DictConfig):
    print(run_export(cfg))


if hydra is None:
    if __name__ == "__main__":
        _main(fallback_cli_config("toy_teacher_export.yaml"))
else:
    main = hydra.main(config_path="configs", config_name="teacher_export", version_base=None)(_main)
    if __name__ == "__main__":
        explicit_config = explicit_config_argument("teacher_export.yaml")
        if explicit_config is not None:
            _main(load_yaml_config(explicit_config))
        else:
            main()
