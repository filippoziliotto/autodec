import os
from pathlib import Path

import torch
from tqdm import tqdm

from autodec.utils.checkpoints import save_autodec_checkpoint
from autodec.visualizations import build_wandb_log


def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def loss_requires_consistency_pass(loss_fn):
    return float(getattr(loss_fn, "lambda_cons", 0.0)) > 0.0


def model_forward(model, points, return_consistency=False):
    if return_consistency:
        return model(points, return_consistency=True)
    return model(points)


class AutoDecTrainer:
    """Device-safe trainer for AutoDec model/loss pairs."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        dataloaders,
        loss_fn,
        ctx,
        device,
        wandb_run=None,
        start_epoch=0,
        best_val_loss=float("inf"),
        is_distributed=False,
        train_sampler=None,
        visualizer=None,
        wandb_visual_log_builder=build_wandb_log,
        visualize_every_n_epochs=None,
        visualize_num_samples=None,
        visualize_split=None,
        log_visualizations_to_wandb=None,
        metric_logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        self.ctx = ctx
        self.device = device
        self.wandb_run = wandb_run
        self.start_epoch = start_epoch
        self.best_val_loss = best_val_loss
        self.is_distributed = is_distributed
        self.train_sampler = train_sampler
        self.visualizer = visualizer
        self.metric_logger = metric_logger
        self.wandb_visual_log_builder = wandb_visual_log_builder
        self.num_epochs = getattr(ctx, "num_epochs", 1)
        self.save_path = getattr(ctx, "save_path", None)
        self.visualize_every_n_epochs = (
            visualize_every_n_epochs
            if visualize_every_n_epochs is not None
            else getattr(ctx, "visualize_every_n_epochs", 1)
        )
        self.visualize_num_samples = (
            visualize_num_samples
            if visualize_num_samples is not None
            else getattr(ctx, "visualize_num_samples", 1)
        )
        self.visualize_split = (
            visualize_split if visualize_split is not None else getattr(ctx, "visualize_split", "val")
        )
        self.log_visualizations_to_wandb = (
            log_visualizations_to_wandb
            if log_visualizations_to_wandb is not None
            else getattr(ctx, "log_visualizations_to_wandb", True)
        )

    def save_checkpoint(self, epoch, val_loss):
        if not is_main_process() or self.save_path is None:
            return None
        filename = f"epoch_{epoch + 1}.pt"
        path = Path(self.save_path) / filename
        return save_autodec_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch,
            val_loss,
            path,
        )

    def _run_loader(self, loader, training, epoch):
        self.model.train(training)
        avg_metrics = {}
        total_batches = 0
        desc = "Train" if training else "Eval"
        pbar = tqdm(loader, desc=f"{desc} {epoch + 1}/{self.num_epochs}", leave=False)

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            for batch in pbar:
                batch = move_batch_to_device(batch, self.device)
                outdict = model_forward(
                    self.model,
                    batch["points"].float(),
                    return_consistency=loss_requires_consistency_pass(self.loss_fn),
                )
                loss, metrics = self.loss_fn(batch, outdict)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                total_batches += 1
                for key, value in metrics.items():
                    avg_metrics[key] = avg_metrics.get(key, 0.0) + float(value)
                pbar.set_postfix({key: f"{value:.4f}" for key, value in metrics.items()})

        if total_batches == 0:
            return avg_metrics
        for key in avg_metrics:
            avg_metrics[key] /= total_batches
        return avg_metrics

    def train_one_epoch(self, epoch):
        if self.is_distributed and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        metrics = self._run_loader(self.dataloaders["train"], training=True, epoch=epoch)
        if self.wandb_run is not None and is_main_process():
            self.wandb_run.log({f"train/{key}": value for key, value in metrics.items()}, step=epoch)
        return metrics

    @torch.no_grad()
    def _log_epoch_visualizations(self, epoch):
        if self.visualizer is None or not is_main_process():
            return []
        if self.visualize_every_n_epochs <= 0:
            return []
        if (epoch + 1) % self.visualize_every_n_epochs != 0:
            return []

        loader = self.dataloaders.get(self.visualize_split)
        if loader is None:
            return []

        was_training = self.model.training
        self.model.eval()
        try:
            batch = next(iter(loader))
            batch = move_batch_to_device(batch, self.device)
            outdict = self.model(batch["points"].float())
            records = self.visualizer.write_epoch(
                batch=batch,
                outdict=outdict,
                epoch=epoch,
                split=self.visualize_split,
                num_samples=self.visualize_num_samples,
            )
            if self.wandb_run is not None and self.log_visualizations_to_wandb and records:
                self.wandb_run.log(self.wandb_visual_log_builder(records), step=epoch)
            return records
        except StopIteration:
            return []
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def evaluate(self, epoch):
        if self.is_distributed and not is_main_process():
            return {}
        metrics = self._run_loader(self.dataloaders["val"], training=False, epoch=epoch)
        if self.wandb_run is not None and is_main_process():
            self.wandb_run.log({f"val/{key}": value for key, value in metrics.items()}, step=epoch)
        self._log_epoch_visualizations(epoch)
        return metrics

    def _learning_rates(self):
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def _log_epoch_metrics(self, epoch, train_metrics, val_metrics, evaluated, val_loss):
        if self.metric_logger is None or not is_main_process():
            return
        self.metric_logger.write(
            {
                "epoch": epoch + 1,
                "epoch_index": epoch,
                "train": train_metrics,
                "val": val_metrics if evaluated else None,
                "evaluated": evaluated,
                "val_loss": float(val_loss),
                "lr": self._learning_rates(),
            }
        )

    def train(self):
        save_every = getattr(self.ctx, "save_every_n_epochs", 1)
        eval_every = getattr(self.ctx, "evaluate_every_n_epochs", 1)
        val_loss = self.best_val_loss
        if self.save_path is not None and is_main_process():
            os.makedirs(self.save_path, exist_ok=True)

        for epoch in range(self.start_epoch, self.num_epochs):
            train_metrics = self.train_one_epoch(epoch)
            do_eval = (epoch + 1) % eval_every == 0 or epoch == self.num_epochs - 1
            val_metrics = None
            if do_eval:
                val_metrics = self.evaluate(epoch)
                val_loss = val_metrics.get("all", val_metrics.get("recon", 0.0))
            self._log_epoch_metrics(
                epoch,
                train_metrics,
                val_metrics,
                do_eval,
                val_loss,
            )
            do_save = (epoch + 1) % save_every == 0 or epoch == self.num_epochs - 1
            if do_save:
                self.save_checkpoint(epoch, val_loss)
