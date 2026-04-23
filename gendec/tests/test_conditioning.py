from pathlib import Path

import torch


def _write_multicat_phase1_root(root, split_to_count):
    from gendec.data.examples import save_teacher_example
    from gendec.data.layout import normalization_stats_path, write_split_manifest
    from gendec.data.normalization import compute_normalization_stats, save_normalization_stats
    from gendec.data.toy_builder import build_toy_example

    root = Path(root)
    categories = ["02691156", "03001627"]
    train_tokens = []
    for split, count in split_to_count.items():
        model_index = []
        for category_id in categories:
            for idx in range(count):
                example = build_toy_example(
                    model_id=f"{category_id}_{split}_{idx:04d}",
                    category_id=category_id,
                    num_points=32,
                )
                save_teacher_example(root, example)
                model_index.append({"category_id": category_id, "model_id": example["model_id"]})
                if split == "train":
                    train_tokens.append(example["tokens_e"])
        write_split_manifest(root, split, model_index)
    save_normalization_stats(normalization_stats_path(root), compute_normalization_stats(torch.stack(train_tokens, dim=0)))


def _write_multicat_phase2_root(root, split_to_count, residual_dim=4):
    from gendec.data.examples import save_teacher_example
    from gendec.data.layout import normalization_stats_path, write_split_manifest
    from gendec.data.normalization import compute_normalization_stats, save_normalization_stats
    from gendec.data.toy_builder import build_toy_phase2_example

    root = Path(root)
    categories = ["02691156", "03001627"]
    train_tokens = []
    for split, count in split_to_count.items():
        model_index = []
        for category_id in categories:
            for idx in range(count):
                example = build_toy_phase2_example(
                    model_id=f"{category_id}_{split}_{idx:04d}",
                    category_id=category_id,
                    num_points=32,
                    residual_dim=residual_dim,
                )
                save_teacher_example(root, example)
                model_index.append({"category_id": category_id, "model_id": example["model_id"]})
                if split == "train":
                    train_tokens.append(example["tokens_ez"])
        write_split_manifest(root, split, model_index)
    save_normalization_stats(normalization_stats_path(root), compute_normalization_stats(torch.stack(train_tokens, dim=0)))


def test_scaffold_dataset_exposes_category_indices_for_multiclass_roots(tmp_path):
    from gendec.data.dataset import ScaffoldTokenDataset

    root = tmp_path / "ShapeNet"
    _write_multicat_phase1_root(root, {"train": 2, "test": 1})

    dataset = ScaffoldTokenDataset(root=root, split="test")
    item = dataset[0]

    assert dataset.num_classes == 2
    assert dataset.category_ids == ["02691156", "03001627"]
    assert item["category_index"].ndim == 0
    assert int(item["category_index"]) in {0, 1}


def test_joint_dataset_exposes_category_indices_for_multiclass_roots(tmp_path):
    from gendec.data.dataset import JointTokenDataset

    root = tmp_path / "ShapeNetPhase2"
    _write_multicat_phase2_root(root, {"train": 2, "test": 1}, residual_dim=4)

    dataset = JointTokenDataset(root=root, split="test")
    item = dataset[0]

    assert dataset.num_classes == 2
    assert dataset.category_ids == ["02691156", "03001627"]
    assert item["category_index"].ndim == 0
    assert int(item["category_index"]) in {0, 1}


def test_phase1_model_accepts_optional_class_conditioning():
    from gendec.models.set_transformer_flow import SetTransformerFlowModel

    model = SetTransformerFlowModel(
        token_dim=15,
        hidden_dim=32,
        n_blocks=2,
        n_heads=4,
        conditioning_enabled=True,
        num_classes=3,
    )
    et = torch.randn(2, 16, 15)
    t = torch.rand(2)
    category_index = torch.tensor([0, 2], dtype=torch.long)

    out = model(et, t, category_index=category_index)

    assert model.conditioning_active is True
    assert out.shape == (2, 16, 15)


def test_phase2_model_accepts_optional_class_conditioning():
    from gendec.models.set_transformer_flow import JointSetTransformerFlowModel

    model = JointSetTransformerFlowModel(
        explicit_dim=15,
        residual_dim=4,
        hidden_dim=32,
        n_blocks=2,
        n_heads=4,
        conditioning_enabled=True,
        num_classes=3,
    )
    et = torch.randn(2, 16, 19)
    t = torch.rand(2)
    category_index = torch.tensor([1, 2], dtype=torch.long)

    v_hat_e, v_hat_z, v_hat = model(et, t, category_index=category_index)

    assert model.conditioning_active is True
    assert v_hat_e.shape == (2, 16, 15)
    assert v_hat_z.shape == (2, 16, 4)
    assert v_hat.shape == (2, 16, 19)
