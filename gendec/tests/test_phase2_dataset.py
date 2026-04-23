import tempfile
from pathlib import Path

import torch


def _write_toy_phase2(root, split="train", num_examples=4):
    from gendec.data.toy_builder import write_toy_phase2_dataset

    write_toy_phase2_dataset(root, split=split, num_examples=num_examples, num_points=256)


def test_joint_token_dataset_loads_and_normalizes():
    from gendec.data.dataset import JointTokenDataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_toy_phase2(tmp)
        ds = JointTokenDataset(root=tmp, split="train")
        assert len(ds) == 4

        item = ds[0]
        assert "tokens_ez" in item
        assert "tokens_e" in item
        assert "tokens_z" in item
        assert "exist" in item
        assert "token_mean" in item
        assert "token_std" in item

        # normalized token dimension is 79
        assert item["tokens_ez"].shape == (16, 79)
        assert item["tokens_e"].shape == (16, 15)
        assert item["tokens_z"].shape == (16, 64)
        assert item["token_mean"].shape == (79,)
        assert item["token_std"].shape == (79,)


def test_joint_token_dataset_residual_dim_property():
    from gendec.data.dataset import JointTokenDataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_toy_phase2(tmp)
        ds = JointTokenDataset(root=tmp, split="train")
        assert ds.residual_dim == 64


def test_joint_token_dataset_tokens_ez_raw_is_unnormalized():
    from gendec.data.dataset import JointTokenDataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_toy_phase2(tmp)
        ds = JointTokenDataset(root=tmp, split="train")
        item = ds[0]

        raw = item["tokens_ez_raw"]
        norm = item["tokens_ez"]
        mean = item["token_mean"]
        std = item["token_std"]

        reconstructed = norm * std.unsqueeze(0) + mean.unsqueeze(0)
        assert torch.allclose(reconstructed, raw, atol=1e-5)


def test_joint_token_dataset_dataloader_batches_correctly():
    from torch.utils.data import DataLoader

    from gendec.data.dataset import JointTokenDataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_toy_phase2(tmp)
        ds = JointTokenDataset(root=tmp, split="train")
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert batch["tokens_ez"].shape == (2, 16, 79)
        assert batch["tokens_e"].shape == (2, 16, 15)
        assert batch["tokens_z"].shape == (2, 16, 64)
