from pathlib import Path

import torch

from gendec.config import cfg_get
from gendec.eval.metrics import MetricAverager
from gendec.losses.path import build_flow_batch
from gendec.sampling import sample_joint_scaffolds, sample_scaffolds
from gendec.tokens import TOKEN_DIM
from gendec.training.ema import ModelEma
from gendec.training.metric_logger import EpochMetricLogger
from gendec.training.runtime_metrics import (
    clean_joint_token_field_mse,
    clean_token_field_mse,
    existence_prediction_metrics,
    residual_norm_metrics,
    sample_joint_scaffold_metrics,
    sample_scaffold_metrics,
    teacher_active_count_metrics,
)
from gendec.training.checkpoints import save_phase1_checkpoint
from gendec.utils.logger import TrainingConsoleLogger


class Phase1Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_dataloader,
        cfg,
        device=None,
        val_dataloader=None,
        scheduler=None,
        stats=None,
        wandb_run=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.stats = stats
        self.cfg = cfg
        self.wandb_run = wandb_run

        self.training_cfg = cfg_get(cfg, "training", cfg_get(cfg, "trainer"))
        self.sampling_cfg = cfg_get(cfg, "sampling", cfg_get(cfg, "sampler"))
        metrics_path = cfg_get(self.training_cfg, "metrics_path", None)
        if metrics_path is None:
            checkpoint_path = Path(cfg_get(self.training_cfg, "checkpoint_path"))
            metrics_path = checkpoint_path.with_suffix(".jsonl")
        self.metric_logger = EpochMetricLogger(metrics_path, append=False)
        self.console_logger = TrainingConsoleLogger(
            disable_tqdm=bool(cfg_get(self.training_cfg, "disable_tqdm", False))
        )

        self.use_amp = bool(cfg_get(self.training_cfg, "amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        ema_decay = float(cfg_get(self.training_cfg, "ema_decay", 0.0) or 0.0)
        self.ema = ModelEma(self.model, decay=ema_decay) if ema_decay > 0.0 else None
        self.grad_clip_norm = cfg_get(self.training_cfg, "grad_clip_norm", None)

    def _move_batch(self, batch):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        return moved

    def _flow_batch(self, batch):
        flow_batch = build_flow_batch(batch["tokens_e"])
        flow_batch["exist"] = batch["exist"]
        flow_batch["token_mean"] = batch["token_mean"][0]
        flow_batch["token_std"] = batch["token_std"][0]
        return flow_batch

    def _batch_metrics(self, batch, flow_batch, v_hat, loss_metrics):
        metrics = {key: value for key, value in loss_metrics.items() if key != "per_sample"}
        metrics.update(clean_token_field_mse(flow_batch, v_hat))
        metrics.update(existence_prediction_metrics(flow_batch, v_hat))
        metrics.update(teacher_active_count_metrics(batch))
        return metrics

    def _run_loader(self, loader, train_mode):
        if loader is None:
            return {}

        averager = MetricAverager()
        self.model.train(mode=train_mode)

        progress = self.console_logger.progress_bar(
            loader,
            desc="Train" if train_mode else "Val",
            leave=False,
        )
        for batch in progress:
            batch = self._move_batch(batch)
            flow_batch = self._flow_batch(batch)
            batch_size = int(batch["tokens_e"].shape[0])

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                v_hat = self.model(flow_batch["Et"], flow_batch["t"])
                loss, loss_metrics = self.loss_fn(flow_batch, v_hat)

            if train_mode:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

            batch_metrics = self._batch_metrics(batch, flow_batch, v_hat, loss_metrics)
            averager.update(batch_metrics, batch_size=batch_size)
            self.console_logger.update_progress_postfix(progress, batch_metrics)

        return averager.compute()

    def _eval_model(self):
        return self.ema.module if self.ema is not None else self.model

    def _sample_metrics(self, epoch):
        if self.stats is None:
            return {}, None

        eval_model = self._eval_model()
        preview_samples = int(cfg_get(self.sampling_cfg, "preview_num_samples", cfg_get(self.sampling_cfg, "num_samples", 4)))
        processed = sample_scaffolds(
            model=eval_model,
            stats=self.stats,
            num_samples=preview_samples,
            token_dim=cfg_get(cfg_get(self.cfg, "model"), "token_dim", 15),
            num_steps=int(cfg_get(self.sampling_cfg, "preview_steps", cfg_get(self.sampling_cfg, "num_steps", 50))),
            exist_threshold=float(cfg_get(self.sampling_cfg, "exist_threshold", 0.5)),
            device=self.device,
        )
        metrics = sample_scaffold_metrics(processed)

        preview_every = int(cfg_get(self.training_cfg, "preview_every", 1))
        preview_path = None
        if preview_every > 0 and ((epoch + 1) % preview_every == 0):
            preview_dir = Path(cfg_get(self.training_cfg, "preview_dir", "gendec/data/previews"))
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / f"epoch_{epoch:04d}_preview.pt"
            torch.save(
                {
                    "tokens": processed["tokens"].detach().cpu(),
                    "exist": processed["exist"].detach().cpu(),
                    "active_mask": processed["active_mask"].detach().cpu(),
                    "preview_points": processed["preview_points"].detach().cpu(),
                },
                preview_path,
            )
        return metrics, preview_path

    def train(self):
        num_epochs = int(cfg_get(self.training_cfg, "num_epochs", 1))
        checkpoint_path = Path(cfg_get(self.training_cfg, "checkpoint_path"))
        best_checkpoint_path = Path(
            cfg_get(self.training_cfg, "best_checkpoint_path", checkpoint_path.with_name("best.pt"))
        )

        best_val = float("inf")
        last_result = None

        for epoch in range(num_epochs):
            train_metrics = self._run_loader(self.train_dataloader, train_mode=True)
            with torch.no_grad():
                eval_model = self._eval_model()
                original_model = self.model
                self.model = eval_model
                try:
                    val_metrics = self._run_loader(self.val_dataloader, train_mode=False)
                    sample_metrics, preview_path = self._sample_metrics(epoch)
                finally:
                    self.model = original_model

            current_val = float(val_metrics.get("all", train_metrics.get("all", float("inf"))))

            save_phase1_checkpoint(
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                loss=current_val,
                path=checkpoint_path,
                ema_model=None if self.ema is None else self.ema.module,
            )

            if current_val <= best_val:
                best_val = current_val
                save_phase1_checkpoint(
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=current_val,
                    path=best_checkpoint_path,
                    ema_model=None if self.ema is None else self.ema.module,
                )

            row = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "samples": sample_metrics,
                "preview_path": None if preview_path is None else str(preview_path),
                "checkpoint_path": str(checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
            }
            self.metric_logger.write(row)
            if self.wandb_run is not None:
                payload = {
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"val/{key}": value for key, value in val_metrics.items()},
                    **{f"samples/{key}": value for key, value in sample_metrics.items()},
                }
                if payload:
                    self.wandb_run.log(payload, step=epoch)
            self.console_logger.print_epoch_summary(
                epoch=epoch,
                num_epochs=num_epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                sample_metrics=sample_metrics,
            )

            last_result = {
                "epoch": epoch,
                "checkpoint_path": str(checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
                "loss": train_metrics.get("all", float("inf")),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "sample_metrics": sample_metrics,
                "preview_path": None if preview_path is None else str(preview_path),
            }

        return last_result or {
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path),
            "loss": float("inf"),
            "train_metrics": {},
            "val_metrics": {},
            "sample_metrics": {},
            "preview_path": None,
        }


class Phase2Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_dataloader,
        cfg,
        device=None,
        val_dataloader=None,
        scheduler=None,
        stats=None,
        wandb_run=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.stats = stats
        self.cfg = cfg
        self.wandb_run = wandb_run

        self.training_cfg = cfg_get(cfg, "training", cfg_get(cfg, "trainer"))
        self.sampling_cfg = cfg_get(cfg, "sampling", cfg_get(cfg, "sampler"))
        metrics_path = cfg_get(self.training_cfg, "metrics_path", None)
        if metrics_path is None:
            checkpoint_path = Path(cfg_get(self.training_cfg, "checkpoint_path"))
            metrics_path = checkpoint_path.with_suffix(".jsonl")
        self.metric_logger = EpochMetricLogger(metrics_path, append=False)
        self.console_logger = TrainingConsoleLogger(
            disable_tqdm=bool(cfg_get(self.training_cfg, "disable_tqdm", False))
        )

        self.use_amp = bool(cfg_get(self.training_cfg, "amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        ema_decay = float(cfg_get(self.training_cfg, "ema_decay", 0.0) or 0.0)
        self.ema = ModelEma(self.model, decay=ema_decay) if ema_decay > 0.0 else None
        self.grad_clip_norm = cfg_get(self.training_cfg, "grad_clip_norm", None)

        model_cfg = cfg_get(cfg, "model")
        self.explicit_dim = int(cfg_get(model_cfg, "explicit_dim", TOKEN_DIM))
        self.residual_dim = int(cfg_get(model_cfg, "residual_dim", 64))

    def _move_batch(self, batch):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        return moved

    def _flow_batch(self, batch):
        flow_batch = build_flow_batch(batch["tokens_ez"])
        flow_batch["exist"] = batch["exist"]
        flow_batch["token_mean"] = batch["token_mean"][0]
        flow_batch["token_std"] = batch["token_std"][0]
        flow_batch["E0"] = batch["tokens_ez"]
        return flow_batch

    def _batch_metrics(self, batch, flow_batch, v_hat_e, v_hat_z, loss_metrics):
        metrics = {key: value for key, value in loss_metrics.items() if key != "per_sample"}
        metrics.update(clean_joint_token_field_mse(flow_batch, v_hat_e, explicit_dim=self.explicit_dim))
        metrics.update(residual_norm_metrics(v_hat_z, flow_batch, explicit_dim=self.explicit_dim))
        metrics.update(teacher_active_count_metrics(batch))
        return metrics

    def _run_loader(self, loader, train_mode):
        if loader is None:
            return {}

        averager = MetricAverager()
        self.model.train(mode=train_mode)

        progress = self.console_logger.progress_bar(
            loader,
            desc="Train" if train_mode else "Val",
            leave=False,
        )
        for batch in progress:
            batch = self._move_batch(batch)
            flow_batch = self._flow_batch(batch)
            batch_size = int(batch["tokens_ez"].shape[0])

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                v_hat_e, v_hat_z, _ = self.model(flow_batch["Et"], flow_batch["t"])
                loss, loss_metrics = self.loss_fn(flow_batch, v_hat_e, v_hat_z)

            if train_mode:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

            batch_metrics = self._batch_metrics(batch, flow_batch, v_hat_e, v_hat_z, loss_metrics)
            averager.update(batch_metrics, batch_size=batch_size)
            self.console_logger.update_progress_postfix(progress, batch_metrics)

        return averager.compute()

    def _eval_model(self):
        return self.ema.module if self.ema is not None else self.model

    def _sample_metrics(self, epoch):
        if self.stats is None:
            return {}, None

        eval_model = self._eval_model()
        preview_samples = int(cfg_get(self.sampling_cfg, "preview_num_samples", cfg_get(self.sampling_cfg, "num_samples", 4)))
        token_dim = self.explicit_dim + self.residual_dim
        processed = sample_joint_scaffolds(
            model=eval_model,
            stats=self.stats,
            num_samples=preview_samples,
            token_dim=token_dim,
            num_steps=int(cfg_get(self.sampling_cfg, "preview_steps", cfg_get(self.sampling_cfg, "num_steps", 50))),
            exist_threshold=float(cfg_get(self.sampling_cfg, "exist_threshold", 0.5)),
            explicit_dim=self.explicit_dim,
            device=self.device,
        )
        metrics = sample_joint_scaffold_metrics(processed)

        preview_every = int(cfg_get(self.training_cfg, "preview_every", 1))
        preview_path = None
        if preview_every > 0 and ((epoch + 1) % preview_every == 0):
            preview_dir = Path(cfg_get(self.training_cfg, "preview_dir", "gendec/data/previews"))
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / f"epoch_{epoch:04d}_preview.pt"
            torch.save(
                {
                    "tokens": processed["tokens"].detach().cpu(),
                    "tokens_z": processed["tokens_z"].detach().cpu(),
                    "exist": processed["exist"].detach().cpu(),
                    "active_mask": processed["active_mask"].detach().cpu(),
                    "preview_points": processed["preview_points"].detach().cpu(),
                },
                preview_path,
            )
        return metrics, preview_path

    def train(self):
        num_epochs = int(cfg_get(self.training_cfg, "num_epochs", 1))
        checkpoint_path = Path(cfg_get(self.training_cfg, "checkpoint_path"))
        best_checkpoint_path = Path(
            cfg_get(self.training_cfg, "best_checkpoint_path", checkpoint_path.with_name("best.pt"))
        )

        best_val = float("inf")
        last_result = None

        for epoch in range(num_epochs):
            train_metrics = self._run_loader(self.train_dataloader, train_mode=True)
            with torch.no_grad():
                eval_model = self._eval_model()
                original_model = self.model
                self.model = eval_model
                try:
                    val_metrics = self._run_loader(self.val_dataloader, train_mode=False)
                    sample_metrics, preview_path = self._sample_metrics(epoch)
                finally:
                    self.model = original_model

            current_val = float(val_metrics.get("all", train_metrics.get("all", float("inf"))))

            save_phase1_checkpoint(
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                loss=current_val,
                path=checkpoint_path,
                ema_model=None if self.ema is None else self.ema.module,
            )

            if current_val <= best_val:
                best_val = current_val
                save_phase1_checkpoint(
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=current_val,
                    path=best_checkpoint_path,
                    ema_model=None if self.ema is None else self.ema.module,
                )

            row = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "samples": sample_metrics,
                "preview_path": None if preview_path is None else str(preview_path),
                "checkpoint_path": str(checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
            }
            self.metric_logger.write(row)
            if self.wandb_run is not None:
                payload = {
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"val/{key}": value for key, value in val_metrics.items()},
                    **{f"samples/{key}": value for key, value in sample_metrics.items()},
                }
                if payload:
                    self.wandb_run.log(payload, step=epoch)
            self.console_logger.print_epoch_summary(
                epoch=epoch,
                num_epochs=num_epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                sample_metrics=sample_metrics,
            )

            last_result = {
                "epoch": epoch,
                "checkpoint_path": str(checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
                "loss": train_metrics.get("all", float("inf")),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "sample_metrics": sample_metrics,
                "preview_path": None if preview_path is None else str(preview_path),
            }

        return last_result or {
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path),
            "loss": float("inf"),
            "train_metrics": {},
            "val_metrics": {},
            "sample_metrics": {},
            "preview_path": None,
        }
