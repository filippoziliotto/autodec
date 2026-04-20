# configs

This folder contains AutoDec-local Hydra configs. They live under `autodec/`
because the current work must not modify the repository-level `configs/`
folder.

Files:

```text
smoke.yaml
train_phase1.yaml
train_phase2.yaml
eval_test.yaml
```

The AutoDec training entrypoint is:

```bash
python -m autodec.training.train
```

The AutoDec test-evaluation entrypoint is:

```bash
python -m autodec.eval.run
```

Both use this folder as their Hydra config path.

## Shared Structure

All configs include:

```text
seed
run_name
dataset
device
use_wandb
wandb
visualization
checkpoints
autodec
shapenet
trainer
optimizer
loss
```

`eval_test.yaml` replaces training-only optimizer/scheduler fields with an
`eval` section and uses `shapenet.categories: null` by default so the
category-balanced visualization selector can cover at least five categories.

### `checkpoints`

Fields:

```text
encoder_from
resume_from
keep_epoch
strict_encoder
```

Use `encoder_from` to initialize the AutoDec encoder from a pretrained SuperDec
checkpoint.

Use `resume_from` to resume a full AutoDec checkpoint.

### `wandb`

WandB is disabled unless:

```text
use_wandb: true
```

When enabled, `autodec.training.train` initializes a WandB run with:

```text
wandb.project
wandb.api_key_env
run_name
```

The AutoDec configs use `wandb.project: autodec` and do not set a WandB entity
or username. Authentication is expected to come from the environment variable
named by `wandb.api_key_env`, which defaults to `WANDB_API_KEY`.

The trainer logs scalar metrics under `train/*` and `val/*`. If
`visualization.log_to_wandb` is also true, it logs the 3D visualization files
as:

```text
visual/gt
visual/sq_mesh
visual/reconstruction
```

Local scalar metrics are also written when:

```yaml
trainer:
  log_metrics_to_file: true
  metrics_log_filename: metrics.jsonl
```

The file lives under the run checkpoint directory, for example
`checkpoints/<run_name>/metrics.jsonl`, and contains one JSON object per epoch
with full `train` and `val` metric dictionaries.

### `visualization`

Controls local 3D visualization export and optional WandB upload.

Fields:

```text
enabled
root_dir
every_n_epochs
num_samples
split
log_to_wandb
mesh_resolution
exist_threshold
max_points
```

When enabled, the trainer writes files once per selected epoch under:

```text
data/viz/<run_name>/<split>/epoch_xxxx/sample_xxxx/
```

The files are:

```text
input_gt.ply
sq_mesh.obj
reconstruction.ply
metadata.json
```

For `eval_test.yaml`, the visualization output root defaults to:

```text
data/eval/<run_name>/<split>/epoch_0000/sample_xxxx/
```

The eval metadata also includes:

```text
category
model_id
dataset_index
checkpoint
metrics
```

### `autodec`

Fields:

```text
residual_dim
primitive_dim
n_surface_samples
exist_tau
encoder
decoder
```

`autodec.encoder` has the same core shape as the old `superdec` config:

```text
decoder.n_queries
decoder.n_layers
decoder.n_heads
decoder.masked_attention
decoder.swapped_attention
decoder.dim_feedforward
decoder.deep_supervision
decoder.pos_encoding_type
point_encoder.l1/l2/l3
```

`autodec.decoder` configures the new neural decoder:

```text
hidden_dim
n_heads
positional_frequencies
component_feature_dim
n_blocks
self_attention_mode
offset_scale
offset_cap
```

`positional_frequencies` controls Fourier features for sampled SQ surface
coordinates. A value of `6` concatenates raw XYZ with
`sin(2^k*pi*x), cos(2^k*pi*x)` for `k = 0..5`.

`component_feature_dim` controls split projections for decoder inputs. If it is
`null`, the decoder uses `max(4, hidden_dim // 4)`. Position features, `E_dec`,
residual `Z`, and the existence gate are projected separately before being
concatenated. Set it to `0` to disable this projection and use the older raw
concatenation path.

`n_blocks` controls how many offset-decoder attention blocks are stacked.
`self_attention_mode: within_primitive` applies self-attention independently to
the sampled points from each primitive before primitive-token cross-attention.
Use `self_attention_mode: none`, `positional_frequencies: 0`, and
`component_feature_dim: 0` for the older single-cross-attention decoder shape.

`offset_cap` enables the primitive-scale offset bound
`tanh(raw_offset) * offset_cap * mean(scale_j)` for points sampled from
primitive `j`. The default YAML value is `0.3`. Set it to `null` to keep the
older unbounded offset behavior. `offset_scale` is the legacy scalar bound and
should only be used when `offset_cap` is `null`.

### `shapenet`

AutoDec ShapeNet runs are configured to train and validate on chairs by
default:

```text
categories: ["03001627"]
```

`03001627` is the ShapeNet synset id for chair. The category filtering is
handled by the existing SuperDec `ShapeNet` dataset.

AutoDec also supports optional deterministic split-size limits:

```text
max_train_items
max_val_items
subset_seed
```

If a max field is `null`, the whole filtered split is used. If it is an integer
smaller than the split length, `autodec.training.builders.limit_dataset` wraps
the dataset in a deterministic `torch.utils.data.Subset`.

Examples:

```bash
python -m autodec.training.train --config-name train_phase1 shapenet.max_train_items=512 shapenet.max_val_items=128
```

or with the script:

```bash
bash autodec/scripts/run_phase1.sh shapenet.max_train_items=512 shapenet.max_val_items=128
```

### `trainer`

Training configs include:

```text
save_path
log_metrics_to_file
metrics_log_filename
save_every_n_epochs
evaluate_every_n_epochs
num_epochs
batch_size
num_workers
augmentations
occlusions
force_occlusions
new_camera_sample
```

`save_path` is combined with `run_name` by `autodec.training.train`, so local
metrics and checkpoints are written under `save_path/run_name/`.

### `optimizer`

Fields:

```text
phase
lr
decoder_lr
encoder_lr
residual_lr
weight_decay
betas
enable_scheduler
```

Phase 1 trains only:

```text
encoder.residual_projector
decoder
```

Phase 2 trains:

```text
encoder backbone
encoder residual projector
decoder
```

### `eval`

Only `eval_test.yaml` has this section.

Fields:

```text
split
output_dir
max_batches
compute_loss_metrics
compute_paper_metrics
f_score_threshold
use_lm_optimization
prune_decoded_points
prune_exist_threshold
prune_target_count
```

`split` defaults to `test`. `output_dir` defaults to `data/eval`, and the
evaluator writes:

```text
data/eval/<run_name>/metrics.json
data/eval/<run_name>/per_sample_metrics.jsonl
```

`compute_loss_metrics` enables the same scalar loss metrics used in training.
`compute_paper_metrics` enables symmetric Chamfer-L1, symmetric Chamfer-L2,
x100-scaled Chamfer values, and F-score metrics for paper-style reporting.
`f_score_threshold` defaults to `0.01`.

`use_lm_optimization` defaults to `false`. When true, standalone test
evaluation enables SuperDec's LM refinement inside `AutoDecEncoder` before the
AutoDec decoder runs. This is intended as an evaluation-only ablation because it
decodes refined SQ parameters with the original residual latents. The current
SuperDec LM implementation uses CUDA-only operations, so this flag requires a
CUDA device.

`prune_decoded_points` enables inference-style pruning before paper metrics and
test visualizations. `prune_exist_threshold` defaults to `0.5`. If
`prune_target_count` is `null`, pruned outputs are resampled to the target point
count from the current batch.

with differential learning rates.

### `loss`

Fields:

```text
type: autodec
phase
lambda_sq
lambda_par
lambda_exist
lambda_cons
n_sq_samples
exist_point_threshold
min_backward_weight
```

Phase 1 should set SQ/parsimony/existence lambdas to zero.
`lambda_cons` defaults to `0.0`. When enabled, it uses a true no-residual
decoder pass, `decoder(E_dec, Z=0)`, rather than raw scaffold Chamfer.

Phase 2 enables:

```text
lambda_sq
lambda_par
lambda_exist
```

## `smoke.yaml`

Smallest intended run config.

Key choices:

```text
P = 4
residual_dim = 16
n_surface_samples = 16
batch_size = 1
num_workers = 0
num_epochs = 1
phase = 1
categories = ["03001627"]
max_train_items = 8
max_val_items = 4
```

Purpose:

Exercise the AutoDec training pipeline with tiny shapes before launching a real
run.

## `train_phase1.yaml`

Decoder warmup config.

Key choices:

```text
encoder_from: checkpoints/shapenet/ckpt.pt
phase: 1
categories: ["03001627"]
lambda_sq: 0
lambda_par: 0
lambda_exist: 0
lambda_cons: 0
```

Expected trainable parameters:

```text
encoder.residual_projector
decoder
```

Frozen parameters:

```text
encoder.point_encoder
encoder.layers
encoder.heads
```

This phase learns to reconstruct from a fixed pretrained primitive
decomposition plus new residual tokens.

## `train_phase2.yaml`

Joint fine-tuning config.

Key choices:

```text
resume_from: checkpoints/autodec_phase1/epoch_200.pt
phase: 2
categories: ["03001627"]
lambda_sq: 1.0
lambda_par: 0.06
lambda_exist: 0.01
lambda_cons: 0.0
```

Expected trainable parameters:

```text
encoder backbone
encoder residual_projector
decoder
```

Uses differential learning rates:

```text
encoder_lr < residual_lr ~= decoder_lr
```

This phase allows the primitive decomposition and decoder to co-adapt while
regularizing the explicit SQ branch.
