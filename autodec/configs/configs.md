# configs

This folder contains AutoDec-local Hydra configs. They live under `autodec/`
because the current work must not modify the repository-level `configs/`
folder.

Files:

```text
smoke.yaml
train_phase1.yaml
train_phase2.yaml
```

The AutoDec training entrypoint is:

```bash
python -m autodec.training.train
```

and uses this folder as its Hydra config path.

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
sq_mesh.glb
reconstruction.ply
metadata.json
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
offset_scale
```

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
min_backward_weight
```

Phase 1 should set SQ/parsimony/existence lambdas to zero.

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
