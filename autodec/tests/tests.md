# tests

This folder contains pytest coverage for the current `autodec/` package. The
test tree mirrors the package tree where useful:

```text
tests/
  test_autodec.py
  test_encoder.py
  test_decoder.py
  models/
  sampling/
  utils/
  losses/
  training/
```

The tests are intentionally small and use injected fake modules where possible.
This keeps tests independent of compiled SuperDec extensions and large model
configs.

## Top-Level Files

### `test_autodec.py`

Tests `autodec.autodec.AutoDec`.

Local fake classes:

```text
ToyEncoder
ToyDecoder
```

`ToyEncoder` exposes the submodule names used by phase-freezing helpers:

```text
point_encoder
layers
heads
residual_projector
```

`ToyDecoder` returns `decoded_points` and `decoded_weights`.

#### `test_autodec_wrapper_runs_encoder_then_decoder_and_exports_attention`

Checks:

```text
model(points, return_attention=True)
```

returns encoder keys plus decoder keys, including:

```text
decoded_points [3, 2, 3]
assign_matrix [3, 5, 1]
decoder_attention [3, 2, 1]
```

#### `test_autodec_phase1_freezes_superdec_backbone_but_not_residual_or_decoder`

Checks `freeze_encoder_backbone()` freezes:

```text
encoder.point_encoder
encoder.layers
encoder.heads
```

but keeps trainable:

```text
encoder.residual_projector
decoder
```

It also verifies `phase1_parameters()` returns exactly residual-projector and
decoder parameters, and `unfreeze_encoder()` restores all parameters to
trainable.

### `test_encoder.py`

Tests `autodec.encoder.AutoDecEncoder`.

Local fake classes:

```text
FakePointEncoder
FakeLayers
FakeHeads
```

These replace the heavy SuperDec point encoder, transformer decoder, and heads.

#### `FakePointEncoder`

Constructor:

```python
FakePointEncoder(feature_dim)
```

Forward input:

```text
points [B, N, 3]
```

Forward output:

```text
point_features [B, N, feature_dim]
```

The tensor is deterministic:

```text
arange(B * N * feature_dim).reshape(B, N, feature_dim)
```

Purpose:

Make assignment-weighted pooling deterministic enough to assert output shapes
without depending on PVCNN.

#### `FakeLayers`

Constructor:

```python
FakeLayers(n_queries, feature_dim)
```

Forward inputs:

```text
init_queries    [P + 1, H]
point_features  [B, N, H]
```

Forward outputs:

```text
refined_queries_list = [queries]
assign_matrices      = [assign_logits]
```

where:

```text
queries       [B, P + 1, H], all ones
assign_logits [B, N, P]
```

The first primitive receives logit `2.0`; the others receive `0.0`. This tests
that `AutoDecEncoder` applies softmax over primitive dimension.

#### `FakeHeads`

Forward input:

```text
sq_features [B, P, H]
```

Forward output:

```text
scale       ones [B, P, 3]
shape       ones [B, P, 2]
rotate      identity matrices [B, P, 3, 3]
trans       zeros [B, P, 3]
exist_logit zeros [B, P, 1]
exist       sigmoid(exist_logit)
```

Purpose:

Provide the exact outdict keys the encoder needs for residual projection.

#### `_ctx`

Builds a minimal `SimpleNamespace` config with:

```text
residual_dim
decoder.n_layers
decoder.n_heads
decoder.n_queries
decoder.deep_supervision
decoder.pos_encoding_type
decoder.dim_feedforward
decoder.swapped_attention
decoder.masked_attention
point_encoder.l3.out_channels
```

This is the minimum shape of `ctx` required by `AutoDecEncoder`.

#### `test_autodec_encoder_returns_superdec_outputs_features_and_residual`

Checks:

```text
assign_matrix shape [2, 5, 2]
point_features shape [2, 5, 4]
sq_features shape [2, 2, 4]
pooled_features shape [2, 2, 4]
residual shape [2, 2, 3]
exist_logit shape [2, 2, 1]
assign_matrix rows sum to 1
```

This proves the encoder emits the original SuperDec-style outputs plus the new
feature/residual outputs required by AutoDec.

### `test_decoder.py`

Tests `autodec.decoder.AutoDecDecoder`.

Local helper:

```text
FixedAngleSampler
_encoder_outdict
```

#### `FixedAngleSampler`

Returns deterministic angles:

```text
etas   [B, P, 2] = 0
omegas [B, P, 2] = [0, pi/2]
```

This avoids using SuperDec's compiled equal-distance sampler during the unit
test.

#### `_encoder_outdict`

Builds a minimal decoder input with:

```text
scale       ones [B, P, 3]
shape       ones [B, P, 2]
trans       zeros [B, P, 3]
rotate      identity [B, P, 3, 3]
exist_logit zeros [B, P, 1]
exist       0.5 [B, P, 1]
residual    random [B, P, residual_dim]
```

#### `test_autodec_decoder_builds_features_and_gates_offsets`

Constructs:

```text
AutoDecDecoder(residual_dim=4, n_surface_samples=2, hidden_dim=16, n_heads=4)
```

Then zeros all offset-decoder parameters. That forces:

```text
decoded_offsets = 0
decoded_points = surface_points
```

Checks:

```text
decoder_features shape [1, 4, 26]
primitive_tokens shape [1, 2, 22]
decoded_offsets shape [1, 4, 3]
decoded_points shape [1, 4, 3]
decoded_points == surface_points
decoded_weights == 0.5
```

Why dimensions are `26` and `22`:

```text
primitive_dim = 18
residual_dim = 4
point feature dim = 3 + 18 + 4 + 1 = 26
primitive token dim = 18 + 4 = 22
```

## Subfolders

### `models/`

Tests neural submodules.

See `autodec/tests/models/models.md`.

### `sampling/`

Tests surface sampling.

See `autodec/tests/sampling/sampling.md`.

### `utils/`

Tests packing helpers.

See `autodec/tests/utils/utils.md`.

### `losses/`

Tests weighted Chamfer, SQ regularizer, and AutoDec loss composition.

See `autodec/tests/losses/losses.md`.

### `training/`

Tests AutoDec training builders and trainer loop.

See `autodec/tests/training/training.md`.
