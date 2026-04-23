# training

## Purpose

`gendec/training/` contains the runtime assembly code for Phase 1 and Phase 2 training: config-driven builders, checkpoint helpers, epoch logging, and the training loops.

## Maintenance Contract

If training construction, checkpoint semantics, or metric logging behavior change, this file must be updated in the same change.

## Files

### `__init__.py`

- Re-export surface for the training layer.
- Exposes `Phase1Trainer`, `Phase2Trainer`, EMA helpers, Phase 1 and Phase 2 builders, config getter, checkpoint helpers, and seed setup.

### `builders.py`

- Config-driven constructors used by train and eval entrypoints.
- `set_seed(seed)`: seeds Python, NumPy, and PyTorch RNGs.
- `_import_wandb()`: lazy WandB import helper so disabled runs do not require the package.
- `build_wandb_run(cfg)`: initializes `wandb.init(project=..., name=run_name)` when `use_wandb` is enabled.
- `_training_cfg(cfg)`: resolves the training section while keeping backward compatibility with the older `trainer` key.
- `_sampling_cfg(cfg)`: resolves the sampling section while keeping backward compatibility with the older `sampler` key.
- `build_dataset(cfg, split=None)`, `build_dataloader(...)`, `build_train_val_dataloaders(cfg)`: Phase 1 dataset and loader builders.
- `build_model(cfg)`, `build_loss(cfg)`: Phase 1 model and loss builders.
- `build_phase2_dataset(cfg, split=None)`, `build_phase2_dataloader(...)`, `build_phase2_train_val_dataloaders(cfg)`: Phase 2 dataset and loader builders.
- `build_phase2_model(cfg)`, `build_phase2_loss(cfg)`: Phase 2 model and loss builders.
- `build_optimizer(cfg, model)`: shared AdamW optimizer builder.
- `build_scheduler(cfg, optimizer, steps_per_epoch)`: shared cosine warmup scheduler builder.

### `checkpoints.py`

- Checkpoint persistence and restore helpers shared by both phases.
- `strip_module_prefix(state_dict)`: removes `module.` prefixes from DDP-style checkpoints.
- `save_phase1_checkpoint(model, optimizer, scheduler, epoch, loss, path, ema_model=None)`: serializes model weights, optional EMA weights, and optimizer/scheduler state.
- `load_phase1_checkpoint(model, path, optimizer=None, scheduler=None, map_location="cpu", load_optimizer=False, use_ema=True)`: restores EMA weights by default when present, and can optionally restore optimizer/scheduler state.

### `ema.py`

- Exponential moving average support.

### `metric_logger.py`

- Append-only JSONL epoch logger.

### `runtime_metrics.py`

- Metric helpers for validation-time diagnostics and training-time unconditional sampling diagnostics.
- `clean_token_field_mse(batch, v_hat)`: Phase 1 explicit per-field MSE.
- `existence_prediction_metrics(batch, v_hat)`: Phase 1 existence entropy and confidence diagnostics.
- `teacher_active_count_metrics(batch, threshold=0.5)`: active-slot stats for teacher tokens.
- `sample_scaffold_metrics(processed, ...)`: Phase 1 sampling validity and active-slot diagnostics.
- `clean_joint_token_field_mse(batch, v_hat_e, explicit_dim=TOKEN_DIM)`: Phase 2 explicit per-field MSE inside the joint-token batch.
- `residual_norm_metrics(v_hat_z, batch, explicit_dim=TOKEN_DIM)`: Phase 2 reconstructed residual-latent norm diagnostics.
- `sample_joint_scaffold_metrics(processed, ...)`: Phase 2 sampling validity and residual diagnostics.

### `schedulers.py`

- Learning-rate scheduler helpers.

### `trainer.py`

- Main training loop implementations.
- `Phase1Trainer`: Phase 1 train/val loop with sampling diagnostics, tqdm console logging, checkpointing, JSONL logging, EMA, and optional WandB logging.
- `Phase2Trainer`: Phase 2 train/val loop with joint-token flow batches, split explicit/residual metrics, joint sampling diagnostics, checkpointing, JSONL logging, EMA, and optional WandB logging.
