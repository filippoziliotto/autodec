# configs

## Purpose

`gendec/configs/` contains YAML runtime presets for export, training, sampling, smoke runs, and evaluation.

## Maintenance Contract

When any config file changes, this file must be updated to reflect the new purpose or key layout.

## Files

### `eval.yaml`

- Main evaluation preset for a trained full-size Phase 1 model.
- `dataset`: points evaluation at `gendec/data/ShapeNet` with `split: test`.
- `model`: full-size model dimensions (`hidden_dim=256`, `n_blocks=6`, `n_heads=8`).
- `loss`: evaluation-time loss settings.
- `checkpoints.resume_from`: checkpoint path to load, defaulting to the validation-best Phase 1 checkpoint.
- `eval`: batch size, generated sample count, and output directory.
- `sampling`: evaluation-time Euler steps and existence threshold used for unconditional sampling during evaluation.
- `visualization`: writes 10 generated SQ visualizations under `data/viz/<run_name>/test/` by default, including `sq_mesh.obj`, `preview_points.ply`, and metadata.
- `autodec_decode`: optional eval-only bridge into a frozen AutoDec decoder for zero-residual coarse-generation plausibility metrics.

### `eval_val.yaml`

- Validation evaluation preset.
- Inherits `eval.yaml`, but switches `dataset.split` to `val`.
- Used by the checked-in validation script so held-out validation and final test runs stay separate.
- Disables generated-SQ visualization writing so the `data/viz/` export is specific to test evaluation.

### `eval_test.yaml`

- Lightweight evaluation preset for smoke-verified small checkpoints.
- Same structure as `eval.yaml`, but uses the small smoke model (`hidden_dim=32`, `n_blocks=2`, `n_heads=4`) and the smoke checkpoint path.
- Also includes a disabled `autodec_decode` block so the eval code path is configurable in smoke-style environments.
- Keeps visualization writing enabled, but lowers mesh resolution and preview point count for faster smoke execution.

### `sample.yaml`

- Sampling-only preset for unconditional generation from a trained checkpoint.
- `dataset.root`: used to locate normalization stats.
- `model`: model architecture to instantiate before checkpoint load.
- `training.best_checkpoint_path`: checkpoint to restore.
- `sampling`: output count, integration steps, existence threshold, and sample output path.

### `smoke.yaml`

- End-to-end smoke preset used for toy export, training, and sampling verification.
- `export`: toy dataset settings, now emitting `train`, `val`, and `test` manifests in one run.
- `dataset`: training split root plus `val_split`.
- `model`: small debug model.
- `loss`, `optimizer`, `scheduler`, `training`: one-epoch fast training setup with validation and preview logging.
- `sampling`: small sampling run configuration.
- `use_wandb: false`: keeps smoke verification self-contained.
- `wandb`: present for schema parity with the main training config.

### `teacher_export.yaml`

- Real teacher-export preset.
- `export.mode: real`: selects the frozen SuperDec export path.
- `dataset_path`: source ShapeNet directory.
- `output_root`: target exported dataset root.
- `splits`: source manifests to mirror into the exported teacher dataset, defaulting to `train`, `val`, and `test`.
- `category_id`: default category restriction, currently chairs.
- `teacher_config`, `teacher_checkpoint`: frozen teacher assets.
- `num_points`, `max_items`: export resolution and optional truncation.

### `toy_teacher_export.yaml`

- Toy dataset export preset for local verification.
- `export.mode: toy`: selects synthetic scaffold generation.
- `output_root`, `splits`, `num_examples`, `num_points`: shape the toy dataset layout and size.

### `train.yaml`

- Main Phase 1 training preset.
- `use_wandb: true`: enables WandB logging by default for the main training run.
- `wandb`: WandB project and API-key environment variable settings. The intended workflow is to source `autodec/keys/keys.sh` or otherwise set `WANDB_API_KEY` before launching training.
- `dataset`: exported dataset root, training split, and validation split.
- `model`: full-size Set Transformer hyperparameters.
- `loss`: flow and existence-loss settings.
- `optimizer`: AdamW hyperparameters, including `eps`.
- `scheduler`: cosine decay with warmup and minimum LR.
- `training`: batch size, workers, epochs, AMP, EMA, gradient clipping, checkpoint paths, metric log path, and preview output directory.
- `sampling`: preview/eval Euler steps plus existence threshold used for training-time sampling diagnostics.
