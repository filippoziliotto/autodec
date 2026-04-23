from pathlib import Path

import torch

from gendec.data.layout import model_dir, scaffold_example_path
from gendec.data.ordering import (
    compute_assignment_mass,
    compute_primitive_volume,
    deterministic_sort_indices,
    reorder_teacher_outputs,
)
from gendec.models.rotation import matrix_to_rot6d
from gendec.tokens import build_joint_tokens, build_scaffold_tokens


def _exist_logit(outdict):
    if "exist_logit" in outdict:
        return outdict["exist_logit"].to(torch.float32)
    exist = outdict["exist"].clamp(1e-6, 1 - 1e-6).to(torch.float32)
    return torch.logit(exist)


def build_teacher_example(outdict, points, model_id, category_id):
    scale = outdict["scale"].to(torch.float32)
    shape = outdict["shape"].to(torch.float32)
    rotate = outdict["rotate"].to(torch.float32)
    trans = outdict["trans"].to(torch.float32)
    assign_matrix = outdict["assign_matrix"].to(torch.float32)
    exist_logit = _exist_logit(outdict)
    exist = torch.sigmoid(exist_logit)
    rot6d = matrix_to_rot6d(rotate)
    mass = compute_assignment_mass(assign_matrix)
    volume = compute_primitive_volume(scale)
    order = deterministic_sort_indices(exist, mass, volume, trans)

    reorder_fields = {
        "scale": scale,
        "shape": shape,
        "rot6d": rot6d,
        "trans": trans,
        "exist_logit": exist_logit,
        "exist": exist,
        "mass": mass,
        "volume": volume,
        "assign_matrix": assign_matrix,
    }

    # Phase 2: also reorder the residual latents when present.
    residual = outdict.get("residual")
    if residual is not None:
        reorder_fields["residual"] = residual.to(torch.float32)

    reordered = reorder_teacher_outputs(reorder_fields, order)

    tokens_e = build_scaffold_tokens(
        reordered["scale"],
        reordered["shape"],
        reordered["trans"],
        reordered["rot6d"],
        reordered["exist_logit"],
    )

    example = {
        "points": points.to(torch.float32),
        "tokens_e": tokens_e.to(torch.float32),
        "exist": reordered["exist"].to(torch.float32),
        "mass": reordered["mass"].to(torch.float32),
        "volume": reordered["volume"].to(torch.float32),
        "category_id": str(category_id),
        "model_id": str(model_id),
    }

    if residual is not None:
        tokens_z = reordered["residual"].to(torch.float32)
        example["tokens_z"] = tokens_z
        example["tokens_ez"] = build_joint_tokens(tokens_e, tokens_z).to(torch.float32)

    return example


def save_teacher_example(root, example):
    path = scaffold_example_path(root, example["category_id"], example["model_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(example, path)
    return path


def load_teacher_example(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)
