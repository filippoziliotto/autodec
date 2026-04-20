# training

This folder contains AutoDec-specific training integration. It reuses SuperDec
datasets and trainer patterns, but keeps all AutoDec-specific logic inside
`autodec/`.

Files:

```text
__init__.py
builders.py
trainer.py
train.py
```

## `__init__.py`

Exports:

```text
AutoDecTrainer
build_dataloaders
build_loss
build_model
build_optimizer
build_scheduler
build_visualizer
build_wandb_run
```

This allows:

```python
from autodec.training import build_model, AutoDecTrainer
```

## `builders.py`

Defines:

```text
cfg_get
set_seed
build_model
build_loss
build_optimizer
build_scheduler
build_visualizer
build_wandb_run
build_dataloaders
limit_dataset
```

### `cfg_get`

Small helper that reads values from either dict-like configs or attribute-style
configs.

```python
cfg_get(cfg, name, default=None)
```

This keeps builders usable with `SimpleNamespace`, `dict`, and Hydra
`DictConfig`.

### `set_seed`

Seeds:

```text
random
numpy
torch
torch.cuda, if available
```

### `build_model`

Builds:

```text
AutoDec(cfg.autodec)
```

If:

```text
cfg.checkpoints.encoder_from
```

is not null, it loads a SuperDec checkpoint into `model.encoder` using:

```text
load_superdec_encoder_checkpoint
```

This is the phase-1 path for initializing AutoDec from a pretrained SuperDec
model.

### `build_loss`

Builds:

```text
AutoDecLoss
```

from `cfg.loss`.

Supported config fields:

```text
phase
lambda_sq / w_sq
lambda_par / w_par
lambda_exist / w_exist
lambda_cons / w_cons
n_sq_samples / n_samples
sq_tau
exist_point_threshold
active_exist_threshold
chamfer_eps
min_backward_weight
```

The aliases exist because older implementation notes used `w_*` names while
the current `AutoDecLoss` constructor uses `lambda_*`.

If `lambda_cons > 0`, the trainer calls the model with
`return_consistency=True`, causing an extra decoder pass with the residual
latent zeroed. With the default `lambda_cons: 0.0`, the extra pass is not
requested.

### `build_optimizer`

Builds a phase-aware Adam optimizer.

Phase is read from:

```text
cfg.optimizer.phase
```

or falls back to:

```text
cfg.loss.phase
```

Phase 1 behavior:

```text
model.freeze_encoder_backbone()
train encoder.residual_projector + decoder only
lr = cfg.optimizer.decoder_lr or cfg.optimizer.lr
```

Phase 2 behavior:

```text
model.unfreeze_encoder()
```

and creates differential learning-rate groups:

```text
encoder backbone   lr = encoder_lr
residual projector lr = residual_lr
decoder            lr = decoder_lr
```

This is important because pretrained SuperDec parameters should usually move
more slowly than the newly initialized AutoDec decoder.

### `build_scheduler`

Returns `None` if:

```text
cfg.optimizer.enable_scheduler == false
```

If enabled, it lazy-imports Hydra and instantiates `cfg.scheduler`.

Hydra is intentionally not imported at module import time, so unit tests and
plain Python imports do not require Hydra to be installed.

### `build_dataloaders`

Reuses SuperDec datasets:

```python
from superdec.data.dataloader import ShapeNet, ABO, ASE_Object
```

Supported dataset names:

```text
shapenet
abo
ase_object
```

Returns:

```text
({"train": train_loader, "val": val_loader}, train_sampler)
```

If distributed training is enabled, it uses `DistributedSampler` for the train
dataset and disables DataLoader shuffle.

For ShapeNet, the dataloader also applies AutoDec-local split-size limits after
constructing the SuperDec dataset:

```text
cfg.shapenet.max_train_items
cfg.shapenet.max_val_items
cfg.shapenet.subset_seed
```

This does not modify `superdec/`. It wraps train and/or validation datasets in
`torch.utils.data.Subset` only when a limit is set and smaller than the filtered
split length.

### `limit_dataset`

Deterministic subset helper:

```python
limit_dataset(dataset, max_items=None, seed=0)
```

Behavior:

```text
max_items is null         -> return the original dataset
max_items >= len(dataset) -> return the original dataset
0 < max_items < len       -> return Subset(dataset, shuffled_indices[:max_items])
```

The shuffled indices are generated from `seed`, so small debugging runs are
repeatable.

### `build_wandb_run`

Initializes a WandB run only when:

```text
cfg.use_wandb == true
```

The import is lazy, so importing `autodec.training.builders` does not require
WandB unless a run is actually requested.

Config fields:

```text
wandb.project
wandb.api_key_env
run_name
```

The run name is passed to WandB as `name`. AutoDec does not pass a WandB
`entity`, so it does not bake a username into the config. Authentication comes
from the environment. By default `wandb.api_key_env` is `WANDB_API_KEY`, which
is the variable consumed by WandB itself.

### `build_visualizer`

Builds:

```text
AutoDecEpochVisualizer
```

when:

```text
cfg.visualization.enabled == true
```

Config fields:

```text
visualization.root_dir
visualization.mesh_resolution
visualization.exist_threshold
visualization.max_points
run_name
```

The visualizer writes local files under `data/viz/` by default and can later be
used to build WandB `Object3D` payloads.

## `trainer.py`

Defines:

```text
AutoDecTrainer
is_main_process
move_batch_to_device
```

### `is_main_process`

Returns true when distributed training is not initialized or the current rank is
zero.

### `move_batch_to_device`

Moves tensor values in a batch dict to the selected device. Non-tensor values
are left unchanged.

### `AutoDecTrainer`

Device-safe AutoDec trainer. Unlike the current root SuperDec trainer, it does
not hardcode `.cuda()` in the training/eval steps.

Constructor inputs:

```text
model
optimizer
scheduler
dataloaders
loss_fn
ctx
device
wandb_run=None
start_epoch=0
best_val_loss=inf
is_distributed=False
train_sampler=None
visualizer=None
wandb_visual_log_builder=build_wandb_log
visualize_every_n_epochs=None
visualize_num_samples=None
visualize_split=None
log_visualizations_to_wandb=None
metric_logger=None
```

Expected dataloaders:

```text
dataloaders["train"]
dataloaders["val"]
```

Expected batch key:

```text
batch["points"] [B, N, 3]
```

Training step:

```text
batch -> device
outdict = model(batch["points"].float())
loss, metrics = loss_fn(batch, outdict)
zero_grad
backward
optimizer.step
scheduler.step, if scheduler exists
```

Evaluation step is the same without gradient updates.

At the end of `evaluate(epoch)`, the trainer can also write and log one
visualization batch. This happens only on the main process and only when a
visualizer is supplied.

Visualization flow:

```text
first batch from dataloaders[visualize_split]
  -> model in eval mode
  -> AutoDecEpochVisualizer.write_epoch(...)
  -> local files in data/viz/<run>/<split>/epoch_xxxx/sample_xxxx/
  -> optional WandB log with visual/gt, visual/sq_mesh, visual/reconstruction
```

The visualization settings are:

```text
visualize_every_n_epochs
visualize_num_samples
visualize_split
log_visualizations_to_wandb
```

When `log_visualizations_to_wandb` is true and `wandb_run` is not null, the
trainer logs the local files as WandB `Object3D`s.

When `metric_logger` is supplied, `train()` writes one JSONL row per epoch after
the optional validation step. Each row contains the 1-based `epoch`, 0-based
`epoch_index`, full `train` metrics, full `val` metrics when evaluated,
`evaluated`, latest `val_loss`, and optimizer learning rates.

Checkpoint saving uses:

```text
save_autodec_checkpoint
```

and writes:

```text
epoch_{epoch + 1}.pt
```

inside `ctx.save_path`.

## `train.py`

Hydra entrypoint for AutoDec training.

Default config path:

```text
autodec/configs
```

Default config:

```text
smoke.yaml
```

Run examples:

```bash
python -m autodec.training.train --config-name smoke
python -m autodec.training.train --config-name train_phase1
python -m autodec.training.train --config-name train_phase2
```

Responsibilities:

1. Choose device and optional DDP rank.
2. Seed RNGs.
3. Build model, loss, dataloaders, optimizer, scheduler.
4. Optionally initialize WandB from `cfg.use_wandb`.
5. Optionally build `AutoDecEpochVisualizer` from `cfg.visualization`.
6. Optionally resume a full AutoDec checkpoint from
   `cfg.checkpoints.resume_from`.
7. Wrap model in DDP if needed.
8. Save the resolved config into the run checkpoint folder.
9. Build `EpochMetricLogger` when `trainer.log_metrics_to_file` is true.
10. Run `AutoDecTrainer.train()`.

The entrypoint imports Hydra directly, so actual CLI use still requires
`hydra-core`.
