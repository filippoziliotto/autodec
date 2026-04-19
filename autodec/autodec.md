# autodec

This folder is the AutoDec package. It is intentionally separate from
`superdec/`: AutoDec imports reusable SuperDec modules, but new or changed
autoencoder behavior lives here.

The current package implements the full composable model path:

```text
AutoDecEncoder -> AutoDecDecoder -> AutoDecLoss
```

`autodec.autodec.AutoDec` now wraps the encoder and decoder so training code can
call a single model:

```python
outdict = model(points)
```

## Public API

`__init__.py` exposes:

```python
from autodec.decoder import AutoDecDecoder
from autodec.encoder import AutoDecEncoder
from autodec.losses import AutoDecLoss
from autodec.models.residual import PartResidualProjector
```

So external code can import:

```python
from autodec import AutoDec, AutoDecEncoder, AutoDecDecoder, AutoDecLoss
```

## Files

### `.gitignore`

Ignores Python bytecode artifacts inside the package:

```text
__pycache__/
*.py[cod]
```

This exists because `autodec/` is currently untracked package work and local
test runs create bytecode directories.

### `__init__.py`

Defines the package-level import surface.

Exports:

```text
AutoDec
AutoDecDecoder
AutoDecEncoder
AutoDecLoss
PartResidualProjector
```

This file does not implement behavior. It only makes the main classes
discoverable from the package root.

### `autodec.py`

Defines:

```text
AutoDec
```

Purpose:

Full model wrapper that composes:

```text
AutoDecEncoder -> AutoDecDecoder
```

Input:

```text
points [B, N, 3]
```

Output:

The decoder-enriched outdict containing primitive fields plus:

```text
decoded_points
decoded_weights
surface_points
decoded_offsets
decoder_features
primitive_tokens
consistency_decoded_points, only when `return_consistency=True`
```

Constructor:

```python
AutoDec(ctx=None, encoder=None, decoder=None)
```

If `encoder` and `decoder` are injected, no config is needed. This is used in
tests and can be used for ablations.

If modules are not injected, `ctx` is expected to contain:

```text
ctx.encoder     SuperDec-like encoder config
ctx.residual_dim
ctx.primitive_dim
ctx.n_surface_samples
ctx.exist_tau
ctx.decoder     AutoDec offset-decoder config
```

The decoder keeps raw sampled SQ coordinates and appends Fourier positional
features before offset prediction. It then applies stacked offset-decoder
blocks: within-primitive self-attention, primitive-token cross-attention, and an
FFN. Set `ctx.decoder.positional_frequencies: 0`,
`ctx.decoder.n_blocks: 1`, and `ctx.decoder.self_attention_mode: none` to
recover the previous shallow decoder shape for checkpoint compatibility.

Training helpers:

```text
freeze_encoder_backbone()
unfreeze_encoder()
phase1_parameters()
encoder_backbone_parameters()
residual_parameters()
decoder_parameters()
```

`freeze_encoder_backbone()` freezes the pretrained SuperDec components:

```text
encoder.point_encoder
encoder.layers
encoder.heads
```

and keeps trainable:

```text
encoder.residual_projector
decoder
```

This is the intended phase-1 training setup.

### `encoder.py`

Defines:

```text
AutoDecEncoder
```

Purpose:

AutoDecEncoder is the SuperDec-compatible encoder plus a per-primitive residual
latent branch. It reuses the SuperDec point encoder and transformer decoder,
then adds:

```text
pooled_features [B, P, H]
residual        [B, P, D]
```

Input:

```text
x [B, N, 3]
```

Output keys required by the decoder:

```text
scale        [B, P, 3]
shape        [B, P, 2]
rotate       [B, P, 3, 3]
trans        [B, P, 3]
exist_logit  [B, P, 1]
exist        [B, P, 1]
residual     [B, P, D]
```

Output keys required by phase-2 losses:

```text
assign_matrix [B, N, P]
```

Optional diagnostic/features keys when `return_features=True`:

```text
point_features  [B, N, H]
sq_features     [B, P, H]
pooled_features [B, P, H]
```

Important implementation details:

- `AutoDecEncoder` imports `TransformerDecoder` and `DecoderLayer` from
  `superdec.models`.
- If no point encoder is injected, it lazy-imports
  `superdec.models.point_encoder.StackedPVConv`.
- The query buffer is `init_queries [P + 1, H]`, matching SuperDec.
- The last query token is dropped before primitive prediction:

  ```python
  sq_features = query_features[:, :-1, ...]
  ```

- Assignment logits are converted to soft assignments using:

  ```python
  outdict["assign_matrix"] = torch.softmax(assign_logits, dim=2)
  ```

  Therefore each point has a distribution over primitive slots:

  ```text
  sum_j assign_matrix[b, i, j] = 1
  ```

- The default head type `"heads"` uses `autodec.models.heads.SuperDecHead`,
  which returns both `exist_logit` and `exist`.
- If a fallback SuperDec MLP head is used and does not return `exist_logit`,
  `_ensure_exist_logit` reconstructs it from `exist` with a clipped `logit`.
- LM optimization exists only as a disabled path:

  ```python
  self.lm_optimization = False
  ```

- The residual branch uses `PartResidualProjector`:

  ```text
  g_j = sum_i M_ij * F_PC_i / (sum_i M_ij + eps)
  z_j = MLP([F_SQ_j, g_j])
  ```

### `decoder.py`

Defines:

```text
AutoDecDecoder
```

Purpose:

AutoDecDecoder decodes primitive parameters and residual latents into a fixed
size point cloud. It implements the Option B decoder: sampled superquadric
surface points plus cross-attention to primitive tokens.

Constructor defaults:

```text
residual_dim       = 64
primitive_dim      = 18
n_surface_samples  = 256
hidden_dim         = 128
n_heads            = 4
exist_tau          = 1.0
positional_frequencies = 6
component_feature_dim = None
n_blocks           = 2
self_attention_mode = within_primitive
offset_scale       = None
```

Input keys:

```text
scale        [B, P, 3]
shape        [B, P, 2]
rotate       [B, P, 3, 3]
trans        [B, P, 3]
exist_logit  [B, P, 1]
residual     [B, P, D]
```

Internal decoder dimensions:

```text
E_dec dim = primitive_dim = 18
position_feature_dim = 3 + 6 * positional_frequencies
component_feature_dim = max(4, hidden_dim // 4) if None
point_feature_dim = 4 * component_feature_dim
primitive_token_dim = 2 * component_feature_dim
```

For default `D=64`, `hidden_dim=128`, and `positional_frequencies=6`:

```text
position_feature_dim = 39
component_feature_dim = 32
point_feature_dim = 128
primitive_token_dim = 64
```

Main steps:

1. `SQSurfaceSampler` samples `S` surface points per primitive.
2. `pack_decoder_primitive_features` builds `E_dec [B, P, 18]`.
3. `repeat_by_part_ids` repeats primitive features and residuals from `[B,P,*]`
   to `[B,P*S,*]`.
4. Per-point decoder features are:

   ```text
   [projected_surface_position, projected_E_dec_for_parent_primitive, projected_residual_for_parent_primitive, projected_gate]
   ```

5. Primitive tokens are:

   ```text
   [projected_E_dec, projected_residual]
   ```

Set `component_feature_dim=0` to disable split projections and recover raw
concatenation:

```text
[surface_position_features, E_dec_for_parent_primitive, residual_for_parent_primitive, gate]
```

6. `CrossAttentionOffsetDecoder` predicts offsets with stacked attention blocks:

   ```text
   decoded_offsets [B, P*S, 3]
   ```

7. Output coordinates are:

   ```text
   decoded_points = surface_points + decoded_weights.unsqueeze(-1) * decoded_offsets
   ```

Important correctness detail:

The existence gate multiplies offsets only. It does not multiply surface
coordinates.

Decoder output keys added to the input outdict:

```text
surface_points            [B, P*S, 3]
surface_points_by_part    [B, P, S, 3]
canonical_surface_points  [B, P, S, 3]
decoded_weights           [B, P*S]
part_ids                  [P*S]
E_dec                     [B, P, 18]
decoder_features          [B, P*S, 3 + 18 + D + 1]
primitive_tokens          [B, P, 18 + D]
decoded_offsets           [B, P*S, 3]
decoded_points            [B, P*S, 3]
consistency_decoded_offsets [B, P*S, 3], only when requested
consistency_decoded_points  [B, P*S, 3], only when requested
decoder_attention         [B, P*S, P], only when requested
```

When `return_consistency=True`, the decoder runs the same sampled SQ surface
through the offset decoder a second time with `residual = 0`. This supports
`lambda_cons > 0` without changing the default forward cost when the consistency
loss is disabled.

## Subfolders

### `models/`

Low-level neural modules:

- primitive prediction head with existence logits
- residual token projection
- cross-attention offset decoder

See `autodec/models/models.md`.

### `sampling/`

Differentiable superquadric surface coordinate construction around the reused
SuperDec equal-distance angle sampler.

See `autodec/sampling/sampling.md`.

### `utils/`

Primitive feature packing and per-surface-point repetition helpers.

See `autodec/utils/utils.md`.

### `losses/`

Weighted reconstruction, sampled SQ regularization, parsimony, existence, and
phase-specific loss composition.

See `autodec/losses/losses.md`.

### `training/`

AutoDec training builders, trainer, and Hydra entrypoint.

See `autodec/training/training.md`.

### `configs/`

AutoDec-local Hydra configs for smoke, phase 1, and phase 2 runs.

See `autodec/configs/configs.md`.

### `tests/`

Pytest coverage mirroring the package structure.

See `autodec/tests/tests.md`.
