from pathlib import Path

import torch

from gendec.config import cfg_get, load_yaml_config
from gendec.data.examples import build_teacher_example, save_teacher_example
from gendec.data.layout import normalization_stats_path, write_split_manifest
from gendec.data.normalization import compute_normalization_stats, save_normalization_stats
from gendec.data.pointclouds import load_source_pointcloud
from gendec.data.shapenet_index import scan_source_shapenet_models
from gendec.data.splits import STANDARD_SHAPENET_SPLITS, resolve_split_names
from gendec.training.checkpoints import strip_module_prefix


def _lazy_load_superdec_model():
    from superdec.superdec import SuperDec

    return SuperDec


def _resolve_teacher_config_path(export_cfg):
    config_path = cfg_get(export_cfg, "teacher_config")
    if config_path is not None:
        return Path(config_path)
    checkpoint_path = Path(cfg_get(export_cfg, "teacher_checkpoint"))
    return checkpoint_path.with_name("config.yaml")


def _teacher_model(export_cfg):
    try:
        SuperDec = _lazy_load_superdec_model()
    except Exception as exc:
        raise RuntimeError(
            "Real teacher export requires the SuperDec model dependencies "
            "to be installed locally."
        ) from exc

    teacher_cfg_path = _resolve_teacher_config_path(export_cfg)
    checkpoint_path = Path(cfg_get(export_cfg, "teacher_checkpoint"))
    if not teacher_cfg_path.is_file():
        raise FileNotFoundError(f"Teacher config not found: {teacher_cfg_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    teacher_cfg = load_yaml_config(teacher_cfg_path)
    model_cfg = cfg_get(teacher_cfg, "superdec", teacher_cfg)
    model = SuperDec(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(strip_module_prefix(state_dict))
    return model


def _output_stats(root, split, token_examples):
    stats_path = normalization_stats_path(root)
    should_write = split in {None, "train"} or not stats_path.is_file()
    if not should_write:
        return stats_path
    if not token_examples:
        raise RuntimeError("Cannot compute normalization stats without exported tokens.")
    stats = compute_normalization_stats(torch.stack(token_examples, dim=0))
    save_normalization_stats(stats_path, stats)
    return stats_path


def _resolved_export_splits(export_cfg):
    return resolve_split_names(
        split=cfg_get(export_cfg, "split", "train"),
        splits=cfg_get(export_cfg, "splits"),
        default="train",
    )


def _export_teacher_split(model, device, export_cfg, output_root, categories, split):
    model_index = []
    token_examples = []
    max_items = cfg_get(export_cfg, "max_items")
    num_points = int(cfg_get(export_cfg, "num_points", 4096))
    source_models = scan_source_shapenet_models(
        cfg_get(export_cfg, "dataset_path", "data/ShapeNet"),
        categories=categories,
        split=split,
        require_manifest=split in STANDARD_SHAPENET_SPLITS,
    )

    with torch.no_grad():
        for item_index, item in enumerate(source_models):
            if max_items is not None and item_index >= int(max_items):
                break

            points = load_source_pointcloud(item["model_dir"], n_points=num_points)["points"]
            outdict = model(points.unsqueeze(0).to(device=device, dtype=torch.float32))
            example = build_teacher_example(
                {
                    key: value[0].detach().cpu() if torch.is_tensor(value) else value
                    for key, value in outdict.items()
                },
                points=points,
                model_id=item["model_id"],
                category_id=item["category_id"],
            )
            save_teacher_example(output_root, example)
            token_examples.append(example["tokens_e"])
            model_index.append(
                {
                    "category_id": item["category_id"],
                    "model_id": item["model_id"],
                }
            )

    if split is not None:
        write_split_manifest(output_root, split, model_index)
    stats_path = _output_stats(output_root, split, token_examples)
    return {
        "root": output_root,
        "split": split,
        "num_examples": len(model_index),
        "normalization_path": stats_path,
    }


def export_teacher_dataset(cfg):
    export_cfg = cfg_get(cfg, "export")
    splits = _resolved_export_splits(export_cfg)
    categories = cfg_get(export_cfg, "categories")
    if categories is None:
        category_id = cfg_get(export_cfg, "category_id")
        categories = None if category_id is None else [category_id]

    output_root = Path(cfg_get(export_cfg, "output_root", "gendec/data/ShapeNet"))
    output_root.mkdir(parents=True, exist_ok=True)

    model = _teacher_model(export_cfg)
    device_name = cfg_get(export_cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model = model.to(device)
    model.eval()

    results = [
        _export_teacher_split(
            model=model,
            device=device,
            export_cfg=export_cfg,
            output_root=output_root,
            categories=categories,
            split=split_name,
        )
        for split_name in splits
    ]

    return {
        "root": output_root,
        "splits": splits,
        "results": results,
        "num_examples": sum(item["num_examples"] for item in results),
        "normalization_path": results[-1]["normalization_path"] if results else normalization_stats_path(output_root),
    }
