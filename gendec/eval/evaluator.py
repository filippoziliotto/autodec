import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gendec.eval.autodec_bridge import build_frozen_autodec_decoder, decode_scaffolds_with_zero_residual
from gendec.eval.metrics import (
    MetricAverager,
    active_decoded_point_count,
    active_primitive_count,
    nearest_neighbor_paper_metrics,
    token_channel_mean_abs,
)
from gendec.losses.path import build_flow_batch
from gendec.sampling import sample_scaffolds
from gendec.training.builders import cfg_get


def _batch_size(batch):
    return int(batch["tokens_e"].shape[0])


class Phase1Evaluator:
    def __init__(self, cfg, model, loss_fn, dataset, device=None):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.device = device or torch.device("cpu")
        eval_cfg = cfg_get(cfg, "eval")
        self.split = cfg_get(cfg_get(cfg, "dataset"), "split", "test")
        self.output_dir = Path(cfg_get(eval_cfg, "output_dir", "gendec/data/eval")) / cfg_get(
            cfg, "run_name", "gendec_eval"
        )
        self.autodec_decode_cfg = cfg_get(cfg, "autodec_decode")
        self._frozen_autodec_decoder = None

    def _loader(self):
        eval_cfg = cfg_get(self.cfg, "eval")
        return DataLoader(
            self.dataset,
            batch_size=cfg_get(eval_cfg, "batch_size", 8),
            shuffle=False,
            num_workers=cfg_get(eval_cfg, "num_workers", 0),
            pin_memory=self.device.type == "cuda",
        )

    def _move_batch(self, batch):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        return moved

    def _write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _autodec_decode_enabled(self):
        return bool(cfg_get(self.autodec_decode_cfg, "enabled", False))

    def _get_frozen_autodec_decoder(self):
        if self._frozen_autodec_decoder is not None:
            return self._frozen_autodec_decoder
        self._frozen_autodec_decoder = build_frozen_autodec_decoder(
            config_path=cfg_get(self.autodec_decode_cfg, "config_path"),
            checkpoint_path=cfg_get(self.autodec_decode_cfg, "checkpoint_path", None),
            device=self.device,
        )
        return self._frozen_autodec_decoder

    def evaluate(self):
        self.model.to(self.device)
        self.model.eval()
        averager = MetricAverager()
        per_sample_rows = []
        total_samples = 0
        reference_points = []
        reference_limit = int(cfg_get(self.autodec_decode_cfg, "reference_limit", 32))
        sampling_cfg = cfg_get(self.cfg, "sampling", cfg_get(self.cfg, "sampler"))

        with torch.no_grad():
            for batch_index, batch in enumerate(self._loader()):
                batch = self._move_batch(batch)
                if self._autodec_decode_enabled() and len(reference_points) < reference_limit:
                    remaining = reference_limit - len(reference_points)
                    for points in batch["points"][:remaining]:
                        reference_points.append(points.detach().cpu())
                flow_batch = build_flow_batch(batch["tokens_e"])
                flow_batch["exist"] = batch["exist"]
                flow_batch["token_mean"] = batch["token_mean"][0]
                flow_batch["token_std"] = batch["token_std"][0]
                v_hat = self.model(flow_batch["Et"], flow_batch["t"])
                _, metrics = self.loss_fn(flow_batch, v_hat, return_per_sample=True)
                batch_size = _batch_size(batch)
                total_samples += batch_size

                averaged_metrics = {key: value for key, value in metrics.items() if key != "per_sample"}
                averager.update(averaged_metrics, batch_size=batch_size)

                per_sample = metrics["per_sample"]
                for sample_index in range(batch_size):
                    per_sample_rows.append(
                        {
                            "batch_index": batch_index,
                            "sample_index": sample_index,
                            "category_id": batch["category_id"][sample_index],
                            "model_id": batch["model_id"][sample_index],
                            "metrics": {
                                key: float(value[sample_index].detach().cpu().item())
                                for key, value in per_sample.items()
                            },
                        }
                    )

            generated = sample_scaffolds(
                model=self.model,
                stats=self.dataset.stats,
                num_samples=cfg_get(cfg_get(self.cfg, "eval"), "generated_num_samples", 4),
                token_dim=cfg_get(cfg_get(self.cfg, "model"), "token_dim", 15),
                num_steps=cfg_get(cfg_get(self.cfg, "eval"), "num_steps", cfg_get(sampling_cfg, "eval_steps", 100)),
                exist_threshold=cfg_get(
                    cfg_get(self.cfg, "eval"),
                    "exist_threshold",
                    cfg_get(sampling_cfg, "exist_threshold", 0.5),
                ),
                device=self.device,
            )

        metrics = averager.compute()
        metrics["generated_active_primitive_count"] = float(
            active_primitive_count(generated["exist"]).detach().cpu().item()
        )
        metrics["generated_token_mean_abs"] = float(
            token_channel_mean_abs(generated["tokens"]).detach().cpu().item()
        )
        metrics["num_rows"] = len(per_sample_rows)

        generated_autodec = None
        if self._autodec_decode_enabled():
            bridge = self._get_frozen_autodec_decoder()
            generated_autodec = decode_scaffolds_with_zero_residual(
                generated,
                decoder=bridge["decoder"],
                residual_dim=bridge["residual_dim"],
            )
            reference_tensor = torch.stack(reference_points, dim=0).to(self.device)
            metrics.update(
                nearest_neighbor_paper_metrics(
                    generated_autodec["surface_points"],
                    reference_tensor,
                    prefix="coarse_surface",
                    point_count=cfg_get(self.autodec_decode_cfg, "point_count", 1024),
                    f_score_threshold=cfg_get(self.autodec_decode_cfg, "f_score_threshold", 0.01),
                )
            )
            metrics.update(
                nearest_neighbor_paper_metrics(
                    generated_autodec["decoded_points"],
                    reference_tensor,
                    prefix="coarse_decoded",
                    point_count=cfg_get(self.autodec_decode_cfg, "point_count", 1024),
                    f_score_threshold=cfg_get(self.autodec_decode_cfg, "f_score_threshold", 0.01),
                )
            )
            metrics["coarse_decoded_active_point_count"] = float(
                active_decoded_point_count(generated_autodec["decoded_weights"]).detach().cpu().item()
            )

        payload = {
            "run_name": cfg_get(self.cfg, "run_name", "gendec_eval"),
            "split": self.split,
            "num_samples": total_samples,
            "metrics": metrics,
        }
        self._write_json(self.output_dir / "metrics.json", payload)
        self._write_jsonl(self.output_dir / "per_sample_metrics.jsonl", per_sample_rows)
        torch.save(
            {
                "tokens": generated["tokens"].cpu(),
                "exist": generated["exist"].cpu(),
                "active_mask": generated["active_mask"].cpu(),
                "preview_points": generated["preview_points"].cpu(),
            },
            self.output_dir / "generated_samples.pt",
        )
        if generated_autodec is not None:
            torch.save(
                {
                    "decoded_points": generated_autodec["decoded_points"].cpu(),
                    "surface_points": generated_autodec["surface_points"].cpu(),
                    "decoded_weights": generated_autodec["decoded_weights"].cpu(),
                    "part_ids": generated_autodec["part_ids"].cpu(),
                },
                self.output_dir / cfg_get(self.autodec_decode_cfg, "output_filename", "generated_autodec_samples.pt"),
            )
        return payload
