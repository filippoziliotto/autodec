from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import yaml


def cfg_get(cfg, name, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def load_yaml_config(path):
    with open(path, encoding="utf-8") as handle:
        return to_namespace(yaml.safe_load(handle))


def fallback_cli_config(default_config_name):
    parser = ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / default_config_name))
    args = parser.parse_args()
    return load_yaml_config(args.config)
