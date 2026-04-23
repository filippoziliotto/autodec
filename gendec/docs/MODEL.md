# Phase 1 Model

This document describes the exact Phase 1 `gendec/` model implemented in this repository as of April 23, 2026.

The Phase 1 system is a generative model over ordered superquadric scaffold tokens. It does not reconstruct point clouds directly. It learns a velocity field in token space and is trained with flow matching.

It supports two runtime modes:

- unconditional generation
- optional class-conditioned generation, enabled only when `conditioning.enabled=true` and `num_classes > 1`

The primary code paths are:

- `gendec/tokens.py`
- `gendec/data/dataset.py`
- `gendec/data/build_teacher_dataset.py`
- `gendec/eval/autodec_bridge.py`
- `gendec/models/components.py`
- `gendec/models/time_embedding.py`
- `gendec/models/set_transformer_flow.py`
- `gendec/losses/path.py`
- `gendec/losses/objectives.py`
- `gendec/losses/flow_matching.py`
- `gendec/sampling.py`
- `gendec/eval/evaluator.py`

## 1. High-Level Objective

The model learns a map

`(E_t, t, c) -> v_theta(E_t, t, c)`

where:

- `E_0` is a clean ordered scaffold token set coming from the SuperDec teacher
- `E_1` is Gaussian noise with the same shape as `E_0`
- `E_t = (1 - t) * E_0 + t * E_1`
- `v_target = E_1 - E_0`
- `v_theta` is the neural network prediction
- `c` is an optional discrete class id

Training minimizes mean squared error between the predicted velocity and the analytic target velocity, with an optional auxiliary existence loss.

At sampling time the model starts from Gaussian noise at `t = 1` and numerically integrates back to `t = 0`. In conditioned runs, the same class id is provided at every Euler step.

## 2. Token Contract

The entire Phase 1 system assumes a fixed token grid:

- number of primitives `P = 16`
- token dimension `D = 15`

This is defined in `gendec/tokens.py`.

Each primitive token has the layout:

1. `scale`: 3 channels
2. `shape`: 2 channels
3. `trans`: 3 channels
4. `rot6d`: 6 channels
5. `exist_logit`: 1 channel

The exact slices are:

- `scale = tokens[..., 0:3]`
- `shape = tokens[..., 3:5]`
- `trans = tokens[..., 5:8]`
- `rot6d = tokens[..., 8:14]`
- `exist_logit = tokens[..., 14:15]`

So one object is represented as:

- `tokens_e`: `[16, 15]`

A minibatch is:

- `tokens_e`: `[B, 16, 15]`

## 3. Teacher Dataset Format

Each exported teacher example is saved at:

`gendec/data/ShapeNet/{category_id}/{model_id}/teacher_scaffold.pt`

The payload keys are:

- `points`: `[4096, 3]`
- `tokens_e`: `[16, 15]`
- `exist`: `[16, 1]`
- `mass`: `[16]`
- `volume`: `[16]`
- `category_id`: `str`
- `model_id`: `str`

The serialized file does not store `category_index` directly. That integer id is resolved by the dataset loader from the category folder names.

Dataset-level normalization is saved at:

`gendec/data/ShapeNet/normalization.pt`

The normalization payload contains:

- `mean`: `[15]`
- `std`: `[15]`

`std` is clamped to at least `1e-6`.

Split manifests live next to each category:

- `gendec/data/ShapeNet/{category_id}/train.lst`
- `gendec/data/ShapeNet/{category_id}/test.lst`
- `gendec/data/ShapeNet/{category_id}/val.lst`

Each manifest contains one `model_id` per line.

The checked-in export config now requests all three source splits in one run:

- `train`
- `val`
- `test`

The exporter resolves `export.splits` or `export.split`. If the value is `all`, it expands to the canonical ordered list above.

## 4. Teacher Export Pipeline

The real export path is implemented in `gendec/data/build_teacher_dataset.py`.

### 4.1 Source inputs

The source ShapeNet layout is assumed to be:

`data/ShapeNet/{category_id}/{model_id}/`

When a concrete split is requested, the exporter reads:

- `data/ShapeNet/{category_id}/train.lst`
- `data/ShapeNet/{category_id}/val.lst`
- `data/ShapeNet/{category_id}/test.lst`

and mirrors those exact membership lists into:

- `gendec/data/ShapeNet/{category_id}/train.lst`
- `gendec/data/ShapeNet/{category_id}/val.lst`
- `gendec/data/ShapeNet/{category_id}/test.lst`

Point clouds are loaded from the first file that exists among:

- `pointcloud_4096.npz`
- `pointcloud.npz`
- `points.npz`

The loader expects an array named `points`.

### 4.2 Point normalization

`gendec/data/pointclouds.py` normalizes points by:

1. computing `translation = mean(points, dim=0)`
2. centering the point cloud
3. computing `scale = 2 * max(abs(centered_points))`
4. dividing the centered points by `scale`

Output shapes:

- raw points: `[N, 3]`
- normalized points: `[4096, 3]`
- translation: `[3]`
- scale: scalar

### 4.3 Teacher forward pass

The export code loads a frozen SuperDec checkpoint and runs:

- input: `[1, 4096, 3]`
- output: a dict containing at least:
  - `scale`: `[1, 16, 3]`
  - `shape`: `[1, 16, 2]`
  - `rotate`: `[1, 16, 3, 3]`
  - `trans`: `[1, 16, 3]`
  - `assign_matrix`: shape from teacher, later reduced to primitive mass
  - either `exist_logit`: `[1, 16, 1]` or `exist`: `[1, 16, 1]`

The export path strips the batch dimension and moves tensors back to CPU before serialization.

### 4.4 Ordering

Teacher outputs are deterministically reordered using `gendec/data/ordering.py`.

The sorting priority is:

1. existence descending
2. assignment mass descending
3. primitive volume descending
4. translation `x` ascending

Inputs to ordering:

- `exist`: `[16, 1]`
- `mass`: `[16]`
- `volume`: `[16]`
- `trans`: `[16, 3]`

Outputs:

- a permutation index `[16]`
- all primitive-dependent tensors reordered consistently

### 4.5 Rotation conversion

The teacher gives a rotation matrix per primitive:

- `rotate`: `[16, 3, 3]`

`gendec/models/rotation.py` converts this to 6D by taking the first two rotation columns:

- `rot6d`: `[16, 6]`

### 4.6 Final token assembly

`gendec/tokens.build_scaffold_tokens` concatenates:

- `scale [16, 3]`
- `shape [16, 2]`
- `trans [16, 3]`
- `rot6d [16, 6]`
- `exist_logit [16, 1]`

into:

- `tokens_e [16, 15]`

## 5. Dataset Loader

`gendec/data/dataset.py` defines `ScaffoldTokenDataset`.

### 5.1 Index building

At construction:

1. it reads `root`
2. it optionally filters by `split`
3. if a split manifest exists for a category, it uses the manifest
4. otherwise it falls back to enumerating category/model directories

The resulting dataset index is a list of dicts with:

- `category_id`
- `model_id`
- `path`

The dataset also builds a stable category vocabulary:

- `category_ids`: sorted list of visible category ids
- `category_to_index`: `category_id -> integer index`
- `num_classes`: size of that vocabulary

### 5.2 Returned sample

`__getitem__` loads one teacher example and returns:

- `points`: `[4096, 3]`
- `tokens_e_raw`: `[16, 15]`
- `tokens_e`: `[16, 15]` normalized
- `exist`: `[16, 1]`
- `mass`: `[16]`
- `volume`: `[16]`
- `token_mean`: `[15]`
- `token_std`: `[15]`
- `category_id`: string
- `category_index`: scalar integer tensor
- `model_id`: string

Normalization is channelwise:

`tokens_e = (tokens_e_raw - mean) / std`

The same `mean` and `std` are returned on every sample because they are dataset-level statistics.

## 6. Flow Path Construction

`gendec/losses/path.py` defines `build_flow_batch`.

Input:

- `e0`: `[B, 16, 15]`
- optional `e1`: `[B, 16, 15]`
- optional `t`: `[B]`

If omitted:

- `e1 ~ N(0, I)` with shape `[B, 16, 15]`
- `t ~ Uniform(0, 1)` with shape `[B]`

Constructed tensors:

- `E0 = e0`: `[B, 16, 15]`
- `E1 = e1`: `[B, 16, 15]`
- `t`: `[B]`
- `t_tokens = t.view(B, 1, 1)`: `[B, 1, 1]`
- `Et = (1 - t_tokens) * E0 + t_tokens * E1`: `[B, 16, 15]`
- `velocity_target = E1 - E0`: `[B, 16, 15]`

No gradients are required through the analytic path construction because `E0`, `E1`, and `Et` are treated as training inputs.

## 7. Neural Network Architecture

The network is defined in `gendec/models/set_transformer_flow.py`.

Default hyperparameters:

- `token_dim = 15`
- `hidden_dim = 256`
- `n_blocks = 6`
- `n_heads = 8`
- `dropout = 0.0`

### 7.1 Full forward signature

Inputs:

- `et`: `[B, 16, 15]`
- `t`: `[B]`
- `category_index`: optional `[B]`

Output:

- `v_hat`: `[B, 16, 15]`

### 7.2 Submodule: TokenProjection

Defined in `gendec/models/components.py`.

Input:

- token tensor `[B, 16, 15]`

Layers:

1. `Linear(15 -> H)`
2. `LayerNorm(H)`
3. `SiLU`
4. `Linear(H -> H)`

Output:

- token features `[B, 16, H]`

With default hyperparameters this is:

- `[B, 16, 256]`

### 7.3 Submodule: SinusoidalTimeEmbedding

Defined in `gendec/models/time_embedding.py`.

Input:

- `t`: `[B]`

Output:

- time embedding `[B, H]`

The module computes a sinusoidal embedding from the scalar time coordinate, then projects it through an MLP to `hidden_dim`.

The forward pass in `SetTransformerFlowModel` immediately unsqueezes it:

- `time_hidden.unsqueeze(1)`: `[B, 1, H]`

This broadcast-adds the same time vector to all 16 primitive tokens.

### 7.4 Time conditioning

After token projection:

- `token_hidden`: `[B, 16, H]`

After time embedding:

- `time_hidden`: `[B, 1, H]`

Conditioned token state:

- `token_hidden = token_hidden + time_hidden`

Output shape remains:

- `[B, 16, H]`

If class conditioning is active, the model also resolves:

- `class_hidden = Embedding(category_index)`: `[B, H]`

and adds it to both:

- primitive token states `[B, 16, H]`
- the learned global token `[B, 1, H]`

If conditioning is disabled or `num_classes <= 1`, this branch is skipped entirely.

### 7.5 Submodule: GlobalToken

Defined in `gendec/models/components.py`.

Learned parameter:

- `self.token`: `[1, 1, H]`

At runtime:

- expanded to `[B, 1, H]`

This token is prepended to the 16 primitive tokens.

### 7.6 Concatenated transformer input

Before attention blocks:

- global token: `[B, 1, H]`
- primitive tokens: `[B, 16, H]`

Concatenation:

- `hidden = cat([global_token, token_hidden], dim=1)`: `[B, 17, H]`

The first token index is the global token. Token indices `1..16` correspond to primitives.

### 7.7 Submodule: SetTransformerBlock

Defined in `gendec/models/components.py`.

Input:

- `hidden`: `[B, 17, H]`

Internal structure:

1. `MultiheadAttention(embed_dim=H, num_heads=n_heads, batch_first=True)`
2. residual add
3. `LayerNorm(H)`
4. feed-forward MLP:
   - `Linear(H -> 4H)`
   - `SiLU`
   - `Linear(4H -> H)`
5. residual add
6. `LayerNorm(H)`

Output:

- `[B, 17, H]`

There are `n_blocks` identical blocks in sequence. With default settings:

- 6 blocks

### 7.8 Information flow inside attention

Every token can attend to every other token because no mask is applied.

That means:

- each primitive token can read all other primitives
- each primitive token can read the global token
- the global token can aggregate information from all primitives

There is no explicit cross-attention, local windowing, or causal ordering. This is permutation-sensitive only because the primitive slots are already sorted offline.

### 7.9 Submodule: VelocityHead

Defined in `gendec/models/components.py`.

After the attention stack:

- `hidden`: `[B, 17, H]`

The global token is dropped:

- `hidden[:, 1:, :]`: `[B, 16, H]`

The velocity head applies:

1. `Linear(H -> H)`
2. `SiLU`
3. `Linear(H -> 15)`

Output:

- `v_hat`: `[B, 16, 15]`

Each primitive token predicts the velocity for its own 15-dimensional scaffold token.

## 8. Loss Function

`gendec/losses/flow_matching.py` defines `FlowMatchingLoss`.

### 8.1 Inputs

The loss expects:

- `batch["Et"]`: `[B, 16, 15]`
- `batch["velocity_target"]`: `[B, 16, 15]`
- `batch["t"]`: `[B]`
- optional `batch["exist"]`: `[B, 16, 1]`
- optional `batch["token_mean"]`: `[15]`
- optional `batch["token_std"]`: `[15]`
- `v_hat`: `[B, 16, 15]`

### 8.2 Flow loss

`gendec/losses/objectives.per_sample_flow_mse` computes:

1. `v_hat - velocity_target`: `[B, 16, 15]`
2. square elementwise
3. reshape to `[B, 240]`
4. mean across the 240 token channels per sample

Output:

- `flow_loss_per_sample`: `[B]`

Then:

- `flow_loss = mean(flow_loss_per_sample)`: scalar

### 8.3 Clean-token reconstruction for auxiliary loss

`gendec/losses/objectives.reconstruct_clean_tokens` computes:

`E0_hat = Et - t * v_hat`

with broadcasted `t`:

- `t_tokens = t.view(B, 1, 1)`: `[B, 1, 1]`
- `E0_hat`: `[B, 16, 15]`

This is the model-implied estimate of the clean scaffold under the straight-line flow parameterization used here.

### 8.4 Existence auxiliary loss

If `exist` is present and `lambda_exist > 0`, `per_sample_exist_bce` runs:

1. reconstruct `E0_hat`: `[B, 16, 15]`
2. select the existence channel:
   - normalized `exist_logit_hat = E0_hat[..., exist_channel]`: `[B, 16]`
3. unnormalize with dataset stats:
   - `logits = exist_logit_hat * std[channel] + mean[channel]`: `[B, 16]`
4. convert target existence to binary:
   - `target = (exist.squeeze(-1) > 0.5)`: `[B, 16]`
5. apply `binary_cross_entropy_with_logits(..., reduction="none")`
6. mean across the 16 primitive slots

Output:

- `exist_loss_per_sample`: `[B]`

### 8.5 Final loss

Per sample:

`all_per_sample = flow_loss_per_sample + lambda_exist * exist_loss_per_sample`

Batch loss:

`loss = mean(all_per_sample)`

Metrics returned:

- `flow_loss`: Python float
- `exist_loss`: Python float if enabled
- `all`: Python float

If `return_per_sample=True`, the loss also returns:

- `per_sample["flow_loss"]`: `[B]`
- `per_sample["exist_loss"]`: `[B]` if enabled
- `per_sample["all"]`: `[B]`

## 9. Backpropagation Path

This section is intentionally explicit.

### 9.1 Flow-loss gradient path

For the main flow loss:

1. `loss` depends on `v_hat`
2. `v_hat` depends on `VelocityHead`
3. `VelocityHead` depends on the final hidden primitive tokens
4. final hidden primitive tokens depend on every `SetTransformerBlock`
5. each block depends on:
   - attention projection weights
   - feed-forward weights
   - layer norm parameters
6. the first block depends on:
   - `TokenProjection`
   - `SinusoidalTimeEmbedding`
   - `GlobalToken`

So gradients flow through:

- velocity head
- all self-attention blocks
- token projection
- time embedding MLP
- learned global token parameter

They do not flow into:

- teacher tokens on disk
- normalization statistics
- the random noise sample `E1`
- the sampled scalar time `t`

### 9.2 Existence-loss gradient path

The existence auxiliary introduces an additional path:

1. `exist_loss` depends on reconstructed `E0_hat`
2. `E0_hat = Et - t * v_hat`
3. therefore `exist_loss` differentiates with respect to `v_hat`
4. the gradient is scaled by `-t`
5. it flows backward through the same network parameters as the flow loss

This means the last token channel is trained in two ways:

- directly by flow matching on the normalized existence-logit channel
- indirectly by BCE on the reconstructed clean existence logits

### 9.3 What receives gradients

Trainable parameters are:

- `TokenProjection` weights and biases
- `TokenProjection` layer norm affine parameters
- `SinusoidalTimeEmbedding` projection parameters
- `GlobalToken.token`
- all attention projections inside every `SetTransformerBlock`
- all feed-forward MLP weights and biases inside every block
- all layer norm affine parameters inside every block
- all `VelocityHead` weights and biases

Non-trainable runtime tensors include:

- `E0`
- `E1`
- `Et`
- `velocity_target`
- normalization statistics
- raw teacher point clouds

## 10. Sampling

Sampling is implemented in `gendec/sampling.py`.

### 10.1 Initial state

The sampler draws:

- `tokens ~ N(0, I)` with shape `[B, 16, 15]`

This is interpreted as the state at `t = 1`.

### 10.2 Time grid

The sampler constructs:

- `time_grid = linspace(1.0, 0.0, num_steps + 1)`

So for `num_steps = 50`, the grid has 51 points.

### 10.3 Euler step

For each interval:

1. current time `t_cur`: `[B]`
2. positive step magnitude `dt = time_grid[i] - time_grid[i + 1]`
3. predict `velocity = model(tokens, t_cur)`: `[B, 16, 15]`
4. integrate backward:

`tokens = tokens - velocity * dt`

Broadcasting uses:

- `dt.view(1, 1, 1)`

Output after the last step:

- sampled normalized clean tokens `[B, 16, 15]`

### 10.4 Post-processing

`postprocess_tokens` does:

1. unnormalize the sampled tokens
2. split channels into semantic fields
3. clamp scale to at least `1e-3`
4. clamp shape exponents into `[0.1, 2.0]`
5. convert 6D rotations to `3 x 3` matrices
6. compute `exist = sigmoid(exist_logit)`
7. threshold existence to get `active_mask`

Shapes:

- raw tokens: `[B, 16, 15]`
- scale: `[B, 16, 3]`
- shape: `[B, 16, 2]`
- trans: `[B, 16, 3]`
- rot6d: `[B, 16, 6]`
- rotate: `[B, 16, 3, 3]`
- exist_logit: `[B, 16, 1]`
- exist: `[B, 16, 1]`
- active_mask: `[B, 16]`

### 10.5 Preview rendering

`render_scaffold_preview` generates superquadric surface points only for visualization.

For each primitive:

1. create an `(eta, omega)` angular grid
2. evaluate the canonical superquadric surface
3. rotate by the primitive rotation matrix
4. translate by the primitive translation
5. keep only active primitives
6. flatten and pad across the batch

Output:

- `preview_points`: `[B, M, 3]`

`M` depends on how many active primitives exist in the batch.

These points are not used in training.

## 11. Evaluation

`gendec/eval/evaluator.py` evaluates a pretrained checkpoint on the exported ShapeNet test split.

### 11.1 Dataset path

The evaluator expects:

- dataset root in `cfg.dataset.root`
- split in `cfg.dataset.split`

For test evaluation this should normally be:

- `root = gendec/data/ShapeNet`
- `split = test`

### 11.2 Per-batch evaluation

For each test batch:

1. load normalized `tokens_e`
2. sample a fresh `E1`
3. sample a fresh `t`
4. build `Et`
5. run the model to predict `v_hat`
6. compute loss metrics
7. store averaged metrics
8. store one JSONL row per sample

This is evaluating how well the pretrained model predicts the teacher-induced flow target on held-out teacher scaffolds.

### 11.3 Generated-sample side output

After loss evaluation, the evaluator also samples unconditional objects from the model and writes:

- `generated_samples.pt`

with:

- `tokens`
- `exist`
- `active_mask`
- `preview_points`

This is not a one-to-one reconstruction metric. It is a side artifact for qualitative and coarse statistical inspection.

### 11.4 Optional Frozen AutoDec Coarse-Decode Evaluation

`gendec` now has an optional eval-only bridge from sampled scaffolds into a frozen AutoDec decoder.

This branch is controlled by:

- `cfg.autodec_decode.enabled`
- `cfg.autodec_decode.config_path`
- `cfg.autodec_decode.checkpoint_path`

When enabled, evaluation performs these extra steps after unconditional scaffold sampling:

1. load a frozen `AutoDecDecoder`
2. convert sampled scaffold fields into the decoder input outdict
3. set `residual = 0` for every primitive
4. decode coarse point clouds with the frozen decoder
5. compare those generated coarse point clouds to a bounded reference subset from the held-out test split using nearest-neighbor paper-style metrics

The decoder-input outdict contains:

- `scale`: `[B, 16, 3]`
- `shape`: `[B, 16, 2]`
- `rotate`: `[B, 16, 3, 3]`
- `trans`: `[B, 16, 3]`
- `exist_logit`: `[B, 16, 1]`
- `exist`: `[B, 16, 1]`
- `residual`: `[B, 16, R]` where `R` is the AutoDec residual dimension and every value is zero

This means the eval branch is explicitly measuring:

`E_hat -> frozen AutoDec decoder with Z=0 -> X_hat_coarse`

not:

`E_hat -> learned residual latent -> X_hat`

So it is a coarse-generation validation path, not a reproduction of the full AutoDec inference stack.

### 11.5 Nearest-Neighbor Coarse Plausibility Metrics

The frozen AutoDec eval branch reports nearest-neighbor metrics against a bounded reference bank from the test split.

For each generated point cloud:

1. subsample or tile it to `point_count`
2. subsample or tile each reference cloud to `point_count`
3. compute paper-style Chamfer/F-score metrics against every reference cloud
4. keep the reference cloud with the lowest Chamfer-L2
5. average the selected metrics over the generated batch

Reported metric families include:

- `coarse_surface_nn_*`: nearest-neighbor plausibility of the raw SQ surface before the AutoDec offset decoder
- `coarse_decoded_nn_*`: nearest-neighbor plausibility after frozen AutoDec decoding with zero residuals
- `coarse_decoded_active_point_count`: average number of active decoded points after existence gating

This is still a plausibility proxy, not a full generative-distribution metric such as MMD or coverage over the whole test split.

## 12. Runtime Entry Points

### 12.1 Teacher export

`gendec/export_teacher.py`

- toy mode: writes synthetic teacher examples into the same on-disk format, and can emit `train`/`val`/`test` in one command
- real mode: runs frozen SuperDec over ShapeNet inputs and serializes scaffold tokens, preserving the source split manifests

### 12.2 Training

`gendec/train.py`

1. build train and val datasets plus dataloaders
2. build model
3. build loss
4. build optimizer
5. build the cosine warmup scheduler
6. train for `num_epochs`
7. run held-out validation after each epoch
8. run unconditional sampling diagnostics after each epoch
9. save `last` and `best` checkpoints
10. append structured epoch metrics to JSONL

The training loop now supports:

- AMP when `training.amp = true`
- gradient clipping with `training.grad_clip_norm`
- EMA with `training.ema_decay`
- validation using `dataset.val_split`
- preview artifact writing with `training.preview_dir`
- optional WandB logging when `use_wandb = true`

When WandB is enabled, `gendec` initializes:

- `project = cfg.wandb.project`
- `name = cfg.run_name`

and logs one payload per epoch with:

- `train/*`
- `val/*`
- `samples/*`

The code reads the active process environment for credentials. The intended shell workflow is to source `autodec/keys/keys.sh` before launching training, so `WANDB_API_KEY` is already present.

Validation metrics include:

- `flow_loss`
- `exist_loss`
- `exist_entropy`
- `exist_confident_fraction`
- `field_mse_scale`
- `field_mse_shape`
- `field_mse_translation`
- `field_mse_rotation6d`
- `field_mse_existence`

Training-time unconditional sampling diagnostics include:

- `active_primitive_count_mean`
- `active_primitive_count_std`
- `positive_scale_fraction`
- `valid_shape_fraction`
- `rotation_orthonormal_fraction`
- `primitive_count_after_threshold_mean`
- `primitive_count_after_threshold_std`

### 12.3 Sampling

`gendec/sample.py`

1. load normalization stats
2. load the configured best checkpoint
3. sample scaffold tokens with Euler integration
4. post-process to superquadric fields
5. save tokens and preview outputs

### 12.4 Evaluation

`gendec/eval/run.py`

1. load checkpoint from `cfg.checkpoints.resume_from` or the configured best-training path
2. load exported test split
3. compute held-out flow/existence metrics
4. sample unconditional scaffold tokens using `sampling.eval_steps`
5. optionally decode sampled scaffolds through a frozen AutoDec decoder with `Z=0`
6. write `metrics.json`
7. write `per_sample_metrics.jsonl`
8. write `generated_samples.pt`
9. on test runs, write 10 generated SQ visualization folders under `data/viz/<run_name>/test/`
10. optionally write `generated_autodec_samples.pt`

Each generated visualization folder contains:

- `sq_mesh.obj`
- `sq_mesh.mtl`
- `preview_points.ply`
- `metadata.json`

## 13. Extension Points

The code is intentionally split so each concern can be changed independently.

### 13.1 To change token layout

Edit:

- `gendec/tokens.py`
- dataset export code
- model `token_dim`
- loss channel assumptions
- sampling post-processing

### 13.2 To change the neural architecture

Edit:

- `gendec/models/components.py`
- `gendec/models/set_transformer_flow.py`

The rest of the training stack only assumes:

- input `[B, 16, 15]`
- time `[B]`
- output `[B, 16, 15]`

### 13.3 To change the flow path

Edit:

- `gendec/losses/path.py`
- `gendec/losses/objectives.py`
- `gendec/losses/flow_matching.py`

### 13.4 To change evaluation

Edit:

- `gendec/eval/evaluator.py`
- `gendec/eval/metrics.py`

## 14. Current Scope

What Phase 1 currently does:

- generates ordered superquadric scaffolds
- trains on teacher token sets
- evaluates held-out token-space flow prediction
- produces previewable superquadric point sets

What Phase 1 intentionally does not do:

- decode residual geometry
- reconstruct dense point clouds directly
- solve primitive permutation online
- optimize test-time latent codes
- run a second-stage AutoDec residual decoder

That separation is deliberate. Phase 1 is only the scaffold prior.
