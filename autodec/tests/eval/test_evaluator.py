import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


class TinyEvalDataset:
    def __init__(self):
        self.models = []
        for category in ["a", "b", "c", "d", "e"]:
            for item_index in range(4):
                self.models.append(
                    {
                        "category": category,
                        "model_id": f"{category}_{item_index}",
                    }
                )

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        return {
            "points": torch.zeros(3, 3) + float(index),
            "idx": index,
            "model_id": self.models[index]["model_id"],
        }


class TinyEvalModel(nn.Module):
    def forward(self, points):
        batch_size = points.shape[0]
        return {
            "decoded_points": points + 0.1,
            "decoded_weights": torch.ones(batch_size, points.shape[1]),
            "surface_points": points,
            "scale": torch.ones(batch_size, 1, 3),
            "shape": torch.ones(batch_size, 1, 2),
            "rotate": torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1),
            "trans": torch.zeros(batch_size, 1, 3),
            "exist": torch.ones(batch_size, 1, 1),
        }


class TinyLoss:
    def __call__(self, batch, outdict):
        return torch.tensor(0.25), {
            "recon": 0.25,
            "active_primitive_count": 1.0,
            "offset_ratio": 0.0,
            "scaffold_chamfer": 0.0,
            "all": 0.25,
        }


class TinyVisualizer:
    def __init__(self, root):
        self.root = Path(root)

    def write_epoch(self, batch, outdict, epoch, split, num_samples):
        from autodec.visualizations.epoch import VisualizationRecord

        records = []
        for sample_index in range(num_samples):
            sample_dir = self.root / split / f"sample_{sample_index:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            input_path = sample_dir / "input_gt.ply"
            sq_mesh_path = sample_dir / "sq_mesh.obj"
            reconstruction_path = sample_dir / "reconstruction.ply"
            metadata_path = sample_dir / "metadata.json"
            input_path.write_text("ply\n", encoding="utf-8")
            sq_mesh_path.write_text("o mesh\n", encoding="utf-8")
            reconstruction_path.write_text("ply\n", encoding="utf-8")
            metadata_path.write_text(json.dumps({"sample_index": sample_index}), encoding="utf-8")
            records.append(
                VisualizationRecord(
                    epoch=epoch,
                    split=split,
                    sample_index=sample_index,
                    sample_dir=sample_dir,
                    input_path=input_path,
                    sq_mesh_path=sq_mesh_path,
                    reconstruction_path=reconstruction_path,
                    metadata_path=metadata_path,
                )
            )
        return records


def _cfg(tmp_path):
    return SimpleNamespace(
        run_name="eval_debug",
        device="cpu",
        checkpoints=SimpleNamespace(resume_from="checkpoint.pt"),
        shapenet=SimpleNamespace(batch_size=4, num_workers=0),
        eval=SimpleNamespace(
            split="test",
            output_dir=str(tmp_path),
            max_batches=None,
            compute_loss_metrics=True,
            compute_paper_metrics=True,
        ),
        visualization=SimpleNamespace(
            enabled=True,
            num_samples=20,
            min_categories=5,
            samples_per_category=4,
            log_to_wandb=False,
        ),
    )


def test_evaluator_writes_metrics_and_twenty_category_visualizations(tmp_path):
    from autodec.eval.evaluator import AutoDecEvaluator

    evaluator = AutoDecEvaluator(
        cfg=_cfg(tmp_path),
        model=TinyEvalModel(),
        loss_fn=TinyLoss(),
        dataset=TinyEvalDataset(),
        visualizer=TinyVisualizer(tmp_path / "viz"),
    )

    result = evaluator.evaluate()

    metrics_path = tmp_path / "eval_debug" / "metrics.json"
    per_sample_path = tmp_path / "eval_debug" / "per_sample_metrics.jsonl"
    assert metrics_path.exists()
    assert per_sample_path.exists()
    assert result["num_visualizations"] == 20
    assert result["visualized_categories"] == ["a", "b", "c", "d", "e"]
    assert "paper_chamfer_l1" in result["metrics"]
    assert "recon" in result["metrics"]

    metadata_paths = sorted((tmp_path / "viz").glob("test/sample_*/metadata.json"))
    assert len(metadata_paths) == 20
    metadata = [json.loads(path.read_text()) for path in metadata_paths]
    assert {item["category"] for item in metadata} == {"a", "b", "c", "d", "e"}
    assert all(item["checkpoint"] == "checkpoint.pt" for item in metadata)

