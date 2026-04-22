import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from autodec.eval.metrics import MetricAverager, paper_chamfer_metrics
from autodec.eval.selectors import select_category_balanced_indices
from autodec.training.builders import cfg_get
from autodec.training.trainer import (
    loss_requires_consistency_pass,
    model_forward,
    move_batch_to_device,
)
from autodec.utils.inference import prune_decoded_points
from autodec.visualizations import build_wandb_log


def _batch_size(batch):
    points = batch["points"]
    return int(points.shape[0])


def _jsonable(value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _batch_value(batch, key, sample_index, default=None):
    value = batch.get(key, default)
    if value is None:
        return default
    if torch.is_tensor(value):
        return _jsonable(value[sample_index])
    if isinstance(value, (list, tuple)):
        return value[sample_index]
    return value


_LM_OUTDICT_KEYS = (
    "assign_matrix",
    "scale",
    "shape",
    "rotate",
    "trans",
    "exist",
    "exist_logit",
)


def _clone_lm_outdict(outdict):
    return {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in outdict.items()
        if key in _LM_OUTDICT_KEYS
    }


class AutoDecEvaluator:
    """Standalone test evaluator for trained AutoDec checkpoints."""

    def __init__(
        self,
        cfg,
        model,
        loss_fn,
        dataset,
        visualizer=None,
        device=None,
        wandb_run=None,
        wandb_visual_log_builder=build_wandb_log,
    ):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.visualizer = visualizer
        self.device = device or torch.device("cpu")
        self.wandb_run = wandb_run
        self.wandb_visual_log_builder = wandb_visual_log_builder

        eval_cfg = cfg_get(cfg, "eval")
        self.split = cfg_get(eval_cfg, "split", "test")
        self.output_dir = (
            Path(cfg_get(eval_cfg, "output_dir", "data/eval"))
            / cfg_get(cfg, "run_name", "autodec_test_eval")
        )

    def _loader(self):
        shapenet_cfg = cfg_get(self.cfg, "shapenet")
        return DataLoader(
            self.dataset,
            batch_size=cfg_get(shapenet_cfg, "batch_size", 8),
            shuffle=False,
            num_workers=cfg_get(shapenet_cfg, "num_workers", 0),
            pin_memory=self.device.type == "cuda",
        )

    def _model_category(self, dataset_index):
        model = self.dataset.models[int(dataset_index)]
        return model.get("category")

    def _write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _row_metrics(self, loss_metrics, paper_metrics):
        metrics = {}
        metrics.update({key: _jsonable(value) for key, value in loss_metrics.items()})
        metrics.update({key: _jsonable(value) for key, value in paper_metrics.items()})
        return metrics

    def _prune_target_count(self, eval_cfg, batch):
        target_count = cfg_get(eval_cfg, "prune_target_count", None)
        if target_count is None:
            return int(batch["points"].shape[1])
        return int(target_count)

    def _maybe_prune_outdict(self, outdict, batch):
        eval_cfg = cfg_get(self.cfg, "eval")
        if not cfg_get(eval_cfg, "prune_decoded_points", False):
            return outdict
        result = dict(outdict)
        result["decoded_points"] = prune_decoded_points(
            outdict,
            exist_threshold=cfg_get(eval_cfg, "prune_exist_threshold", 0.5),
            target_count=self._prune_target_count(eval_cfg, batch),
        )
        return result

    def _model_encoder(self):
        model = getattr(self.model, "module", self.model)
        return getattr(model, "encoder", None)

    def _visualization_lm_optimizer(self):
        encoder = self._model_encoder()
        optimizer = getattr(encoder, "lm_optimizer", None)
        if optimizer is not None:
            return optimizer
        if self.device.type != "cuda":
            raise RuntimeError(
                "visualization.write_lm_optimized_sq_mesh requires CUDA unless "
                "a test LM optimizer is injected on model.encoder.lm_optimizer."
            )
        if encoder is None:
            raise TypeError("LM mesh export requires a model with AutoDecEncoder")

        from superdec.lm_optimization.lm_optimizer import LMOptimizer

        optimizer = LMOptimizer()
        if hasattr(optimizer, "to"):
            optimizer = optimizer.to(self.device)
        return optimizer

    def _should_write_lm_sq_mesh(self):
        vis_cfg = cfg_get(self.cfg, "visualization")
        return (
            self.split == "test"
            and cfg_get(vis_cfg, "write_lm_optimized_sq_mesh", False)
        )

    def _lm_optimized_outdict(self, outdict, batch):
        if not self._should_write_lm_sq_mesh():
            return None
        optimizer = self._visualization_lm_optimizer()
        return optimizer(_clone_lm_outdict(outdict), batch["points"].float())

    def _evaluate_loader(self):
        eval_cfg = cfg_get(self.cfg, "eval")
        compute_loss = cfg_get(eval_cfg, "compute_loss_metrics", True)
        compute_paper = cfg_get(eval_cfg, "compute_paper_metrics", True)
        f_score_threshold = cfg_get(eval_cfg, "f_score_threshold", 0.01)
        max_batches = cfg_get(eval_cfg, "max_batches")

        averager = MetricAverager()
        per_sample_rows = []
        total_samples = 0

        self.model.eval()
        for batch_index, batch in enumerate(tqdm(self._loader(), desc=f"Eval {self.split}", leave=False)):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch_to_device(batch, self.device)
            outdict = model_forward(
                self.model,
                batch["points"].float(),
                return_consistency=compute_loss
                and loss_requires_consistency_pass(self.loss_fn),
            )
            batch_size = _batch_size(batch)
            total_samples += batch_size

            loss_metrics = {}
            if compute_loss:
                _, loss_metrics = self.loss_fn(batch, outdict)
                averager.update(loss_metrics, batch_size=batch_size)

            paper_metrics = {}
            if compute_paper:
                paper_outdict = self._maybe_prune_outdict(outdict, batch)
                paper_metrics = paper_chamfer_metrics(
                    paper_outdict["decoded_points"],
                    batch["points"],
                    f_score_threshold=f_score_threshold,
                )
                averager.update(paper_metrics, batch_size=batch_size)

            row_metrics = self._row_metrics(loss_metrics, paper_metrics)
            for sample_index in range(batch_size):
                dataset_index = _batch_value(batch, "idx", sample_index, total_samples)
                per_sample_rows.append(
                    {
                        "batch_index": batch_index,
                        "sample_index": sample_index,
                        "dataset_index": int(dataset_index),
                        "category": self._model_category(dataset_index),
                        "model_id": _batch_value(batch, "model_id", sample_index),
                        "metrics": row_metrics,
                    }
                )

        return averager.compute(), per_sample_rows, total_samples

    def _visualization_selection(self):
        vis_cfg = cfg_get(self.cfg, "visualization")
        samples_per_category = cfg_get(vis_cfg, "samples_per_category", 2)
        return select_category_balanced_indices(
            self.dataset,
            samples_per_category=samples_per_category,
        )

    def _write_visualizations(self, metrics):
        vis_cfg = cfg_get(self.cfg, "visualization")
        if self.visualizer is None or not cfg_get(vis_cfg, "enabled", True):
            return []
        selection = self._visualization_selection()
        items = [self.dataset[item.dataset_index] for item in selection]
        batch = default_collate(items)
        batch = move_batch_to_device(batch, self.device)
        outdict = self.model(batch["points"].float())
        lm_outdict = self._lm_optimized_outdict(outdict, batch)
        outdict = self._maybe_prune_outdict(outdict, batch)
        write_kwargs = {
            "batch": batch,
            "outdict": outdict,
            "epoch": 0,
            "split": self.split,
            "num_samples": len(selection),
        }
        if lm_outdict is not None:
            write_kwargs["lm_outdict"] = lm_outdict
        records = self.visualizer.write_epoch(**write_kwargs)

        checkpoint_cfg = cfg_get(self.cfg, "checkpoints")
        checkpoint = cfg_get(checkpoint_cfg, "resume_from")
        for record, selected in zip(records, selection):
            metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
            metadata.update(
                {
                    "category": selected.category,
                    "model_id": selected.model_id,
                    "dataset_index": selected.dataset_index,
                    "checkpoint": checkpoint,
                    "metrics": metrics,
                }
            )
            self._write_json(record.metadata_path, metadata)

        if (
            self.wandb_run is not None
            and cfg_get(vis_cfg, "log_to_wandb", False)
            and records
        ):
            self.wandb_run.log(
                self.wandb_visual_log_builder(records, prefix=f"{self.split}_visual"),
            )
        return records

    @torch.no_grad()
    def evaluate(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metrics, per_sample_rows, total_samples = self._evaluate_loader()
        records = self._write_visualizations(metrics)

        summary = {
            "run_name": cfg_get(self.cfg, "run_name", "autodec_test_eval"),
            "split": self.split,
            "num_samples": total_samples,
            "num_visualizations": len(records),
            "visualized_categories": sorted(
                {
                    json.loads(record.metadata_path.read_text(encoding="utf-8"))["category"]
                    for record in records
                }
            ),
            "metrics": metrics,
        }
        self._write_json(self.output_dir / "metrics.json", summary)
        self._write_jsonl(self.output_dir / "per_sample_metrics.jsonl", per_sample_rows)
        if self.wandb_run is not None:
            self.wandb_run.log({f"{self.split}/{key}": value for key, value in metrics.items()})
        return summary
