import hashlib
from pathlib import Path

import torch

from gendec.data.examples import save_teacher_example
from gendec.data.layout import model_dir, normalization_stats_path, write_split_manifest
from gendec.data.normalization import compute_normalization_stats, save_normalization_stats
from gendec.data.splits import resolve_split_names
from gendec.tokens import PRIMITIVE_COUNT, RESIDUAL_DIM_DEFAULT, build_joint_tokens, build_scaffold_tokens


def _rotation_6d():
    return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)


def _stable_seed(model_id, category_id):
    digest = hashlib.sha1(f"{category_id}:{model_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)


def build_toy_example(model_id, category_id="03001627", num_points=4096, n_primitives=PRIMITIVE_COUNT):
    generator = torch.Generator().manual_seed(_stable_seed(model_id, category_id))
    active = 6
    points = torch.randn(num_points, 3, generator=generator) * 0.15

    scale = torch.full((n_primitives, 3), 0.05)
    scale[:active] += torch.rand(active, 3, generator=generator) * 0.25
    shape = torch.full((n_primitives, 2), 0.6)
    shape[:active] += torch.rand(active, 2, generator=generator) * 0.6
    trans = torch.zeros(n_primitives, 3)
    trans[:active] = torch.randn(active, 3, generator=generator) * 0.3
    rot6d = _rotation_6d().repeat(n_primitives, 1)
    exist_logit = torch.full((n_primitives, 1), -4.0)
    exist_logit[:active] = 4.0
    tokens_e = build_scaffold_tokens(scale, shape, trans, rot6d, exist_logit)
    exist = torch.sigmoid(exist_logit)
    mass = torch.linspace(1.0, 0.1, n_primitives)
    volume = scale.prod(dim=-1)
    return {
        "points": points.to(torch.float32),
        "tokens_e": tokens_e.to(torch.float32),
        "exist": exist.to(torch.float32),
        "mass": mass.to(torch.float32),
        "volume": volume.to(torch.float32),
        "category_id": category_id,
        "model_id": model_id,
    }


def build_toy_phase2_example(
    model_id,
    category_id="03001627",
    num_points=4096,
    n_primitives=PRIMITIVE_COUNT,
    residual_dim=RESIDUAL_DIM_DEFAULT,
):
    """Build a Phase 2 toy example that includes residual tokens Z alongside E."""
    base = build_toy_example(
        model_id=model_id,
        category_id=category_id,
        num_points=num_points,
        n_primitives=n_primitives,
    )
    generator = torch.Generator().manual_seed(_stable_seed(model_id + "_z", category_id))
    tokens_z = torch.randn(n_primitives, residual_dim, generator=generator) * 0.1
    tokens_ez = build_joint_tokens(base["tokens_e"], tokens_z)
    base["tokens_z"] = tokens_z.to(torch.float32)
    base["tokens_ez"] = tokens_ez.to(torch.float32)
    return base


def write_toy_teacher_dataset(root, split="train", num_examples=8, num_points=4096):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    examples = []
    model_index = []
    model_dirs = []
    for idx in range(num_examples):
        example = build_toy_example(
            model_id=f"toy_{split}_{idx:04d}",
            category_id="03001627",
            num_points=num_points,
        )
        save_teacher_example(root, example)
        examples.append(example["tokens_e"])
        model_dirs.append(model_dir(root, example["category_id"], example["model_id"]))
        model_index.append(
            {
                "category_id": example["category_id"],
                "model_id": example["model_id"],
            }
        )

    if split is not None:
        write_split_manifest(root, split, model_index)

    stats_path = normalization_stats_path(root)
    should_write_stats = split in {None, "train"} or not stats_path.is_file()
    if should_write_stats:
        stats = compute_normalization_stats(torch.stack(examples, dim=0))
        save_normalization_stats(stats_path, stats)

    return {
        "root": root,
        "model_dirs": model_dirs,
        "normalization_path": stats_path,
        "num_examples": num_examples,
        "split": split,
    }


def write_toy_phase2_dataset(
    root,
    split="train",
    num_examples=8,
    num_points=4096,
    residual_dim=RESIDUAL_DIM_DEFAULT,
):
    """Write a toy Phase 2 dataset that stores joint (E, Z) tokens."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    ez_examples = []
    model_index = []
    model_dirs_list = []
    for idx in range(num_examples):
        example = build_toy_phase2_example(
            model_id=f"toy_{split}_{idx:04d}",
            category_id="03001627",
            num_points=num_points,
            residual_dim=residual_dim,
        )
        save_teacher_example(root, example)
        ez_examples.append(example["tokens_ez"])
        model_dirs_list.append(model_dir(root, example["category_id"], example["model_id"]))
        model_index.append(
            {
                "category_id": example["category_id"],
                "model_id": example["model_id"],
            }
        )

    if split is not None:
        write_split_manifest(root, split, model_index)

    stats_path = normalization_stats_path(root)
    should_write_stats = split in {None, "train"} or not stats_path.is_file()
    if should_write_stats:
        stats = compute_normalization_stats(torch.stack(ez_examples, dim=0))
        save_normalization_stats(stats_path, stats)

    return {
        "root": root,
        "model_dirs": model_dirs_list,
        "normalization_path": stats_path,
        "num_examples": num_examples,
        "split": split,
    }


def write_toy_teacher_dataset_splits(root, split=None, splits=None, num_examples=8, num_points=4096):
    resolved_splits = resolve_split_names(split=split, splits=splits, default="train")
    results = []
    for split_name in resolved_splits:
        results.append(
            write_toy_teacher_dataset(
                root=root,
                split=split_name,
                num_examples=num_examples,
                num_points=num_points,
            )
        )

    return {
        "root": Path(root),
        "splits": resolved_splits,
        "results": results,
        "num_examples": sum(item["num_examples"] for item in results),
        "normalization_path": results[-1]["normalization_path"] if results else normalization_stats_path(root),
    }


def write_toy_phase2_dataset_splits(root, split=None, splits=None, num_examples=8, num_points=4096, residual_dim=RESIDUAL_DIM_DEFAULT):
    resolved_splits = resolve_split_names(split=split, splits=splits, default="train")
    results = []
    for split_name in resolved_splits:
        results.append(
            write_toy_phase2_dataset(
                root=root,
                split=split_name,
                num_examples=num_examples,
                num_points=num_points,
                residual_dim=residual_dim,
            )
        )

    return {
        "root": Path(root),
        "splits": resolved_splits,
        "results": results,
        "num_examples": sum(item["num_examples"] for item in results),
        "normalization_path": results[-1]["normalization_path"] if results else normalization_stats_path(root),
    }
