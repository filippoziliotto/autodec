from pathlib import Path

import torch


def test_toy_teacher_dataset_round_trips_schema_and_normalization(tmp_path):
    from gendec.data.dataset import ScaffoldTokenDataset
    from gendec.data.toy_builder import write_toy_teacher_dataset

    root = tmp_path / "ShapeNet"
    write_toy_teacher_dataset(root=root, num_examples=4, num_points=32)

    dataset = ScaffoldTokenDataset(root=root)
    item = dataset[0]

    assert len(dataset) == 4
    assert item["tokens_e"].shape == (16, 15)
    assert item["tokens_e_raw"].shape == (16, 15)
    assert item["points"].shape == (32, 3)
    assert item["exist"].shape == (16, 1)
    assert item["mass"].shape == (16,)
    assert item["volume"].shape == (16,)
    assert set(item) >= {"category_id", "model_id", "tokens_e", "tokens_e_raw"}
    assert (root / "03001627" / item["model_id"] / "teacher_scaffold.pt").is_file()
    assert (root / "normalization.pt").is_file()


def test_saved_example_payload_matches_expected_keys():
    from gendec.data.toy_builder import build_toy_example

    example = build_toy_example(model_id="chair_0001", category_id="03001627", num_points=16)

    assert set(example) == {
        "points",
        "tokens_e",
        "exist",
        "mass",
        "volume",
        "category_id",
        "model_id",
    }
    assert example["tokens_e"].shape == (16, 15)


def test_dataset_scans_category_and_model_directories(tmp_path):
    from gendec.data.dataset import ScaffoldTokenDataset
    from gendec.data.toy_builder import write_toy_teacher_dataset

    root = tmp_path / "ShapeNet"
    if root.exists():
        import shutil

        shutil.rmtree(root)

    try:
        write_toy_teacher_dataset(root=root, num_examples=3, num_points=24)
        dataset = ScaffoldTokenDataset(root=root)
        model_dirs = {(item["category_id"], item["model_id"]) for item in dataset}

        assert len(model_dirs) == 3
        for category_id, model_id in model_dirs:
            assert (root / category_id / model_id / "teacher_scaffold.pt").is_file()
    finally:
        if root.exists():
            import shutil

            shutil.rmtree(root)


def test_dataset_uses_split_manifest_when_requested(tmp_path):
    from gendec.data.dataset import ScaffoldTokenDataset
    from gendec.data.toy_builder import write_toy_teacher_dataset

    root = tmp_path / "ShapeNet"
    write_toy_teacher_dataset(root=root, split="train", num_examples=2, num_points=24)
    write_toy_teacher_dataset(root=root, split="test", num_examples=1, num_points=24)

    dataset = ScaffoldTokenDataset(root=root, split="test")

    assert len(dataset) == 1
    assert dataset[0]["model_id"] == "toy_test_0000"


def test_test_split_export_keeps_existing_train_normalization_stats(tmp_path):
    from gendec.data.normalization import load_normalization_stats
    from gendec.data.toy_builder import write_toy_teacher_dataset

    root = tmp_path / "ShapeNet"
    write_toy_teacher_dataset(root=root, split="train", num_examples=2, num_points=24)
    stats_before = load_normalization_stats(root / "normalization.pt")

    write_toy_teacher_dataset(root=root, split="test", num_examples=1, num_points=24)
    stats_after = load_normalization_stats(root / "normalization.pt")

    assert torch.equal(stats_before["mean"], stats_after["mean"])
    assert torch.equal(stats_before["std"], stats_after["std"])


def test_toy_export_can_materialize_train_val_test_splits_in_one_run(tmp_path):
    from gendec.data.dataset import ScaffoldTokenDataset
    from gendec.export_teacher import run_export

    root = tmp_path / "ShapeNet"
    cfg = {
        "export": {
            "mode": "toy",
            "output_root": str(root),
            "splits": ["train", "val", "test"],
            "num_examples": 2,
            "num_points": 24,
        }
    }

    result = run_export(cfg)

    assert set(result["splits"]) == {"train", "val", "test"}
    assert (root / "03001627" / "train.lst").is_file()
    assert (root / "03001627" / "val.lst").is_file()
    assert (root / "03001627" / "test.lst").is_file()
    assert len(ScaffoldTokenDataset(root=root, split="train")) == 2
    assert len(ScaffoldTokenDataset(root=root, split="val")) == 2
    assert len(ScaffoldTokenDataset(root=root, split="test")) == 2


def test_toy_export_all_keyword_matches_train_val_test_splits(tmp_path):
    from gendec.data.dataset import ScaffoldTokenDataset
    from gendec.export_teacher import run_export

    root = tmp_path / "ShapeNet"
    cfg = {
        "export": {
            "mode": "toy",
            "output_root": str(root),
            "split": "all",
            "num_examples": 1,
            "num_points": 24,
        }
    }

    result = run_export(cfg)

    assert result["splits"] == ["train", "val", "test"]
    assert len(ScaffoldTokenDataset(root=root, split="train")) == 1
    assert len(ScaffoldTokenDataset(root=root, split="val")) == 1
    assert len(ScaffoldTokenDataset(root=root, split="test")) == 1
