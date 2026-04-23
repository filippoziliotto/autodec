# training

## Purpose

`gendec/training/` contains the runtime assembly code for Phase 1 training: config-driven builders, checkpoint helpers, epoch logging, and the training loop.

## Maintenance Contract

If training construction, checkpoint semantics, or metric logging behavior change, this file must be updated in the same change.

## Files

### `__init__.py`

- Re-export surface for the training layer.
- Exposes `Phase1Trainer`, EMA helpers, dataset/model/loss/optimizer/scheduler builders, config getter, checkpoint helpers, and seed setup.

### `builders.py`

- Config-driven constructors used by train and eval entrypoints.
- `set_seed(seed)`: seeds Python, NumPy, and PyTorch RNGs.
- `_import_wandb()`: lazy WandB import helper so disabled runs do not require the package.
- `build_wandb_run(cfg)`: initializes `wandb.init(project=..., name=run_name)` when `use_wandb` is enabled and the API key is already present in the environment.
- `_training_cfg(cfg)`: resolves the training section while keeping backward compatibility with the older `trainer` key.
- `_sampling_cfg(cfg)`: resolves the sampling section while keeping backward compatibility with the older `sampler` key.
- `build_dataset(cfg, split=None)`: constructs `ScaffoldTokenDataset`.
- `build_dataloader(cfg, split=None, batch_size=None, shuffle=None)`: constructs the dataloader with sensible defaults based on split.
- `build_train_val_dataloaders(cfg)`: builds both train and val dataloaders from the exported teacher dataset.
- `build_model(cfg)`: instantiates `SetTransformerFlowModel` from config values.
- `build_loss(cfg)`: instantiates `FlowMatchingLoss`.
- `build_optimizer(cfg, model)`: instantiates AdamW with config values, including `eps`.
- `build_scheduler(cfg, optimizer, steps_per_epoch)`: instantiates the cosine warmup scheduler when configured.

### `checkpoints.py`

- Checkpoint persistence and restore helpers.
- `strip_module_prefix(state_dict)`: removes `module.` prefixes from DDP-style checkpoints.
- `save_phase1_checkpoint(model, optimizer, scheduler, epoch, loss, path, ema_model=None)`: serializes raw model weights, optional EMA weights, and optimizer/scheduler state.
- `load_phase1_checkpoint(model, path, optimizer=None, scheduler=None, map_location="cpu", load_optimizer=False, use_ema=True)`: restores EMA weights by default when present, and can optionally restore optimizer/scheduler state.

### `ema.py`

- Exponential moving average support for Phase 1 checkpoints and validation.
- `ModelEma`:
  - `__init__(model, decay=0.999)`: clones the model into a frozen EMA copy.
  - `update(model)`: updates EMA weights after each optimizer step.
  - `state_dict()`: returns EMA weights for checkpointing.
  - `load_state_dict(state_dict)`: restores EMA weights.

### `metric_logger.py`

- Append-only JSONL epoch logger.
- `_jsonable(value)`: converts nested values into JSON-safe representations.
- `EpochMetricLogger`:
  - `__init__(path, append=True)`: prepares the log file and optionally truncates it.
  - `write(row)`: appends one JSON row.

### `runtime_metrics.py`

- Metric helpers for validation-time diagnostics and training-time unconditional sampling diagnostics.
- `_unnormalize_pair(batch, v_hat)`: reconstructs clean tokens and maps predictions/targets back to raw token units.
- `clean_token_field_mse(batch, v_hat)`: reports per-field MSE over scale, shape, translation, rotation-6D, and existence logit.
- `existence_prediction_metrics(batch, v_hat)`: reports existence entropy and confident-existence fraction.
- `teacher_active_count_metrics(batch, threshold=0.5)`: reports active-primitive count statistics for teacher tokens.
- `sample_scaffold_metrics(processed, valid_shape_range=(0.1, 2.0), orthonormal_tol=1e-3)`: reports sample validity and active-slot statistics after unconditional sampling and postprocessing.

### `schedulers.py`

- Learning-rate scheduler helpers.
- `build_cosine_warmup_scheduler(optimizer, total_steps, warmup_steps=0, min_lr=1e-5)`: returns a per-step cosine-decay scheduler with linear warmup and a floor LR.

### `trainer.py`

- Main training loop implementation.
- `Phase1Trainer`:
  - `__init__(model, loss_fn, optimizer, train_dataloader, cfg, device=None, val_dataloader=None, scheduler=None, stats=None, wandb_run=None)`: stores runtime objects, resolves the device, enables optional AMP/EMA, initializes epoch metric logging, and keeps an optional WandB run handle.
  - `_move_batch(batch)`: moves tensor fields in a batch onto the target device.
  - `_flow_batch(batch)`: builds the straight-line flow batch from one normalized teacher batch.
  - `_batch_metrics(batch, flow_batch, v_hat, loss_metrics)`: combines loss metrics with fieldwise errors, existence entropy, and teacher active-slot stats.
  - `_run_loader(loader, train_mode)`: shared train/val pass with optional optimizer, scheduler, and EMA updates.
  - `_eval_model()`: chooses EMA weights when available for validation and sampling diagnostics.
  - `_sample_metrics(epoch)`: runs unconditional sampling, computes active-count and validity metrics, and writes scaffold preview artifacts on schedule.
  - `train()`: iterates epochs, performs train and val passes, logs unconditional sample diagnostics, saves last/best checkpoints, appends structured epoch rows, and mirrors epoch metrics to WandB when enabled.
