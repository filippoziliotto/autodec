import json
import sys
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


class ConsistencyEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.return_consistency_flags = []

    def forward(self, points, return_consistency=False):
        self.return_consistency_flags.append(return_consistency)
        batch_size = points.shape[0]
        outdict = {
            "decoded_points": points + 0.1,
            "decoded_weights": torch.ones(batch_size, points.shape[1]),
            "surface_points": points,
            "scale": torch.ones(batch_size, 1, 3),
            "shape": torch.ones(batch_size, 1, 2),
            "rotate": torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1),
            "trans": torch.zeros(batch_size, 1, 3),
            "exist": torch.ones(batch_size, 1, 1),
        }
        if return_consistency:
            outdict["consistency_decoded_points"] = points + 0.2
        return outdict


class PrunableEvalModel(nn.Module):
    def forward(self, points):
        batch_size = points.shape[0]
        active = points[:, 0]
        inactive = active + 100.0
        return {
            "decoded_points": torch.stack([active, inactive], dim=1),
            "decoded_weights": torch.ones(batch_size, 2, dtype=points.dtype, device=points.device),
            "surface_points": torch.stack([active, inactive], dim=1),
            "part_ids": torch.tensor([0, 1], device=points.device),
            "scale": torch.ones(batch_size, 2, 3, dtype=points.dtype, device=points.device),
            "shape": torch.ones(batch_size, 2, 2, dtype=points.dtype, device=points.device),
            "rotate": torch.eye(3, dtype=points.dtype, device=points.device)
            .view(1, 1, 3, 3)
            .repeat(batch_size, 2, 1, 1),
            "trans": torch.zeros(batch_size, 2, 3, dtype=points.dtype, device=points.device),
            "exist": torch.tensor([[[0.9], [0.1]]], dtype=points.dtype, device=points.device)
            .repeat(batch_size, 1, 1),
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


class ConsistencyEvalLoss:
    lambda_cons = 1.0

    def __call__(self, batch, outdict):
        assert "consistency_decoded_points" in outdict
        return torch.tensor(0.25), {
            "recon": 0.25,
            "consistency_loss": 0.5,
            "all": 0.75,
        }


class TinyLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimpleNamespace(enable_calls=0)

        def enable_lm_optimization():
            self.encoder.enable_calls += 1

        self.encoder.enable_lm_optimization = enable_lm_optimization


class RecordingEvalLMOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, outdict, points):
        self.calls += 1
        result = dict(outdict)
        result["trans"] = outdict["trans"] + 1.0
        return result


class LMMeshEvalModel(TinyEvalModel):
    def __init__(self):
        super().__init__()
        self.lm_optimizer = RecordingEvalLMOptimizer()
        self.encoder = SimpleNamespace(lm_optimizer=self.lm_optimizer)


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
            f_score_threshold=0.01,
            prune_decoded_points=False,
            prune_exist_threshold=0.5,
            prune_target_count=None,
        ),
        visualization=SimpleNamespace(
            enabled=True,
            samples_per_category=2,
            log_to_wandb=False,
        ),
    )


def test_eval_dataset_builder_applies_shapenet_category_split(monkeypatch):
    from autodec.utils.shapenet_categories import PAPER_UNSEEN_CATEGORIES
    from autodec.eval.run import _build_dataset

    calls = []

    class FakeShapeNet:
        def __init__(self, split, cfg):
            calls.append((split, list(cfg.shapenet.categories)))

    fake_module = SimpleNamespace(ShapeNet=FakeShapeNet)
    monkeypatch.setitem(sys.modules, "superdec.data.dataloader", fake_module)
    cfg = SimpleNamespace(
        dataset="shapenet",
        shapenet=SimpleNamespace(
            category_split="paper_unseen",
            categories=None,
        ),
        eval=SimpleNamespace(split="test"),
    )

    _build_dataset(cfg)

    assert calls == [("test", PAPER_UNSEEN_CATEGORIES)]


def test_evaluator_writes_metrics_and_two_visualizations_per_category(tmp_path):
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
    assert result["num_visualizations"] == 10
    assert result["visualized_categories"] == ["a", "b", "c", "d", "e"]
    assert "paper_chamfer_l1" in result["metrics"]
    assert "paper_full_chamfer_l1" in result["metrics"]
    assert "paper_sq_chamfer_l1" in result["metrics"]
    assert result["metrics"]["paper_chamfer_l1"] == result["metrics"]["paper_full_chamfer_l1"]
    assert result["metrics"]["paper_sq_chamfer_l1"] == 0.0
    assert "paper_f_score_tau_0_01" in result["metrics"]
    assert "recon" in result["metrics"]

    metadata_paths = sorted((tmp_path / "viz").glob("test/sample_*/metadata.json"))
    assert len(metadata_paths) == 10
    metadata = [json.loads(path.read_text()) for path in metadata_paths]
    assert {item["category"] for item in metadata} == {"a", "b", "c", "d", "e"}
    assert all(item["checkpoint"] == "checkpoint.pt" for item in metadata)


def test_evaluator_uses_configured_fscore_threshold(tmp_path):
    from autodec.eval.evaluator import AutoDecEvaluator

    cfg = _cfg(tmp_path)
    cfg.eval.compute_loss_metrics = False
    cfg.eval.f_score_threshold = 0.03
    cfg.visualization.enabled = False
    evaluator = AutoDecEvaluator(
        cfg=cfg,
        model=TinyEvalModel(),
        loss_fn=TinyLoss(),
        dataset=TinyEvalDataset(),
        visualizer=None,
    )

    result = evaluator.evaluate()

    assert "paper_f_score_tau_0_03" in result["metrics"]
    assert "paper_f_score_tau_0_01" not in result["metrics"]


def test_evaluator_computes_paper_metrics_on_pruned_decoded_points(tmp_path):
    from autodec.eval.evaluator import AutoDecEvaluator

    cfg = _cfg(tmp_path)
    cfg.eval.compute_loss_metrics = False
    cfg.eval.prune_decoded_points = True
    cfg.eval.prune_target_count = 1
    cfg.visualization.enabled = False
    evaluator = AutoDecEvaluator(
        cfg=cfg,
        model=PrunableEvalModel(),
        loss_fn=TinyLoss(),
        dataset=TinyEvalDataset(),
        visualizer=None,
    )

    result = evaluator.evaluate()

    assert result["metrics"]["paper_chamfer_l1"] == 0.0
    assert result["metrics"]["paper_chamfer_l2"] == 0.0


def test_evaluator_computes_sq_paper_metrics_on_pruned_surface_points(tmp_path):
    from autodec.eval.evaluator import AutoDecEvaluator

    cfg = _cfg(tmp_path)
    cfg.eval.compute_loss_metrics = False
    cfg.eval.prune_decoded_points = True
    cfg.eval.prune_target_count = 1
    cfg.visualization.enabled = False
    evaluator = AutoDecEvaluator(
        cfg=cfg,
        model=PrunableEvalModel(),
        loss_fn=TinyLoss(),
        dataset=TinyEvalDataset(),
        visualizer=None,
    )

    result = evaluator.evaluate()

    assert result["metrics"]["paper_sq_chamfer_l1"] == 0.0
    assert result["metrics"]["paper_sq_chamfer_l2"] == 0.0


def test_evaluator_requests_consistency_pass_when_loss_needs_it(tmp_path):
    from autodec.eval.evaluator import AutoDecEvaluator

    cfg = _cfg(tmp_path)
    cfg.visualization.enabled = False
    model = ConsistencyEvalModel()
    evaluator = AutoDecEvaluator(
        cfg=cfg,
        model=model,
        loss_fn=ConsistencyEvalLoss(),
        dataset=TinyEvalDataset(),
        visualizer=None,
    )

    result = evaluator.evaluate()

    assert model.return_consistency_flags
    assert all(model.return_consistency_flags)
    assert result["metrics"]["consistency_loss"] == 0.5


def test_evaluator_sends_pruned_decoded_points_to_visualizer(tmp_path):
    from autodec.eval.evaluator import AutoDecEvaluator

    class CaptureVisualizer:
        def __init__(self):
            self.decoded_shape = None

        def write_epoch(self, batch, outdict, epoch, split, num_samples):
            self.decoded_shape = tuple(outdict["decoded_points"].shape)
            return TinyVisualizer(tmp_path / "viz").write_epoch(
                batch,
                outdict,
                epoch,
                split,
                num_samples,
            )

    cfg = _cfg(tmp_path)
    cfg.eval.prune_decoded_points = True
    cfg.eval.prune_target_count = 1
    cfg.visualization.samples_per_category = 1
    visualizer = CaptureVisualizer()
    evaluator = AutoDecEvaluator(
        cfg=cfg,
        model=PrunableEvalModel(),
        loss_fn=TinyLoss(),
        dataset=TinyEvalDataset(),
        visualizer=visualizer,
    )

    evaluator.evaluate()

    assert visualizer.decoded_shape == (5, 1, 3)


def test_evaluator_passes_lm_outdict_to_visualizer_only_for_test_split(tmp_path):
    from autodec.eval.evaluator import AutoDecEvaluator

    class CaptureLMVisualizer:
        def __init__(self):
            self.outdict = None
            self.lm_outdict = None

        def write_epoch(self, batch, outdict, epoch, split, num_samples, lm_outdict=None):
            self.outdict = outdict
            self.lm_outdict = lm_outdict
            return TinyVisualizer(tmp_path / "viz").write_epoch(
                batch,
                outdict,
                epoch,
                split,
                num_samples,
            )

    cfg = _cfg(tmp_path)
    cfg.eval.compute_loss_metrics = False
    cfg.eval.compute_paper_metrics = False
    cfg.visualization.samples_per_category = 1
    cfg.visualization.write_lm_optimized_sq_mesh = True
    model = LMMeshEvalModel()
    visualizer = CaptureLMVisualizer()
    evaluator = AutoDecEvaluator(
        cfg=cfg,
        model=model,
        loss_fn=TinyLoss(),
        dataset=TinyEvalDataset(),
        visualizer=visualizer,
    )

    evaluator.evaluate()

    assert model.lm_optimizer.calls == 1
    assert torch.allclose(visualizer.outdict["trans"], torch.zeros(1, 1, 3))
    assert visualizer.outdict["decoded_points"].shape == (5, 3, 3)
    assert torch.allclose(visualizer.outdict["decoded_points"][0], torch.full((3, 3), 0.1))
    assert torch.allclose(visualizer.lm_outdict["trans"], torch.ones(1, 1, 3))

    cfg.eval.split = "val"
    model = LMMeshEvalModel()
    visualizer = CaptureLMVisualizer()
    evaluator = AutoDecEvaluator(
        cfg=cfg,
        model=model,
        loss_fn=TinyLoss(),
        dataset=TinyEvalDataset(),
        visualizer=visualizer,
    )

    evaluator.evaluate()

    assert model.lm_optimizer.calls == 0
    assert visualizer.lm_outdict is None


def test_eval_lm_optimization_flag_is_disabled_by_default(tmp_path):
    from autodec.eval.run import maybe_enable_lm_optimization

    cfg = _cfg(tmp_path)
    model = TinyLMModel()

    enabled = maybe_enable_lm_optimization(model, cfg, torch.device("cpu"))

    assert enabled is False
    assert model.encoder.enable_calls == 0


def test_eval_lm_optimization_flag_requires_cuda(tmp_path):
    from autodec.eval.run import maybe_enable_lm_optimization

    cfg = _cfg(tmp_path)
    cfg.eval.use_lm_optimization = True

    try:
        maybe_enable_lm_optimization(TinyLMModel(), cfg, torch.device("cpu"))
    except RuntimeError as exc:
        assert "requires CUDA" in str(exc)
    else:
        raise AssertionError("Expected CUDA guard to reject LM optimization on CPU")


def test_eval_rejects_global_lm_when_writing_separate_lm_mesh(tmp_path):
    from autodec.eval.run import maybe_enable_lm_optimization

    cfg = _cfg(tmp_path)
    cfg.eval.use_lm_optimization = True
    cfg.visualization.write_lm_optimized_sq_mesh = True

    try:
        maybe_enable_lm_optimization(TinyLMModel(), cfg, torch.device("cuda"))
    except ValueError as exc:
        assert "eval.use_lm_optimization" in str(exc)
        assert "write_lm_optimized_sq_mesh" in str(exc)
    else:
        raise AssertionError("Expected mutually exclusive LM modes to be rejected")


def test_eval_lm_optimization_flag_enables_encoder_on_cuda(tmp_path):
    from autodec.eval.run import maybe_enable_lm_optimization

    cfg = _cfg(tmp_path)
    cfg.eval.use_lm_optimization = True
    model = TinyLMModel()

    enabled = maybe_enable_lm_optimization(model, cfg, torch.device("cuda"))

    assert enabled is True
    assert model.encoder.enable_calls == 1
