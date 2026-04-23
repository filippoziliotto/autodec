from pathlib import Path

import torch


def _write_example(root, category_id, model_id):
    model_dir = root / category_id / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "points": torch.zeros(4, 3),
            "tokens_e": torch.zeros(16, 15),
            "exist": torch.zeros(16, 1),
            "mass": torch.zeros(16),
            "volume": torch.zeros(16),
            "category_id": category_id,
            "model_id": model_id,
        },
        model_dir / "teacher_scaffold.pt",
    )


def test_iter_exported_examples_uses_split_manifest_when_present(tmp_path):
    from gendec.data.layout import iter_exported_examples

    root = tmp_path / "ShapeNet"
    _write_example(root, "03001627", "chair_a")
    _write_example(root, "03001627", "chair_b")
    (root / "03001627" / "test.lst").write_text("chair_b\n", encoding="utf-8")

    examples = list(iter_exported_examples(root, split="test"))

    assert len(examples) == 1
    assert examples[0]["category_id"] == "03001627"
    assert examples[0]["model_id"] == "chair_b"
    assert examples[0]["path"] == root / "03001627" / "chair_b" / "teacher_scaffold.pt"


def test_scan_source_shapenet_models_falls_back_to_directory_listing(tmp_path):
    from gendec.data.shapenet_index import scan_source_shapenet_models

    root = tmp_path / "ShapeNet"
    (root / "02691156" / "model_a").mkdir(parents=True)
    (root / "02691156" / "model_b").mkdir(parents=True)
    (root / "02691156" / ".DS_Store").write_text("", encoding="utf-8")

    models = scan_source_shapenet_models(root, categories=["02691156"], split="test")

    assert [item["model_id"] for item in models] == ["model_a", "model_b"]
