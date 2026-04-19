# utils

This folder tests `autodec/utils/`.

Files:

```text
test_checkpoints.py
test_inference.py
test_metrics.py
test_packing.py
```

## `test_checkpoints.py`

Tests:

```text
autodec.utils.checkpoints.load_superdec_encoder_checkpoint
autodec.utils.checkpoints.load_autodec_checkpoint
```

### `test_load_superdec_encoder_checkpoint_strips_module_prefix`

Creates a tiny `nn.Linear`, saves a checkpoint whose keys are prefixed with:

```text
module.
```

Then loads the checkpoint into a target module.

Checks:

```text
target.state_dict() == expected.state_dict()
```

Purpose:

Protect loading SuperDec checkpoints saved under DDP into AutoDec's encoder.

### `test_load_autodec_checkpoint_restores_model_and_returns_epoch`

Saves a full checkpoint with:

```text
model_state_dict
epoch
val_loss
```

Loads it into a target model and checks:

```text
epoch == 7
val_loss == 0.25
model weights restored
```

Purpose:

Protect full AutoDec resume behavior.

## `test_inference.py`

Tests:

```text
autodec.utils.inference.prune_decoded_points
```

Checks:

```text
inactive primitive decoded points are removed
fixed-count resampling is deterministic
if no primitive is active, the highest-existence primitive is kept
```

## `test_metrics.py`

Tests:

```text
offset_ratio
active_primitive_count
active_decoded_point_count
primitive_mass_entropy
scaffold_vs_decoded_chamfer
```

### `test_offset_ratio_uses_raw_offsets_against_scaffold_norm`

Uses scaffold points with mean norm `3.5` and offsets with mean norm `3.5`.

Checks:

```text
offset_ratio == 1.0
```

### `test_active_counts_and_assignment_entropy`

Checks:

```text
active_primitive_count([0.9, 0.1, 0.8]) == 2
active_decoded_point_count([0.9, 0.9, 0.1, 0.8]) == 3
primitive_mass_entropy(one-hot primitive mass) == 0
```

### `test_scaffold_vs_decoded_chamfer_reports_both_values`

Uses a target at the origin, scaffold point away from origin, and decoded point
at the origin.

Checks:

```text
decoded_chamfer < scaffold_chamfer
```

## `test_packing.py`

Tests:

```text
autodec.utils.packing.pack_decoder_primitive_features
autodec.utils.packing.pack_serialized_primitive_features
autodec.utils.packing.repeat_by_part_ids
```

## Local Helper

### `_outdict`

Signature:

```python
_outdict(batch=2, primitives=3)
```

Returns:

```text
scale          ones [B, P, 3]
shape          0.5 [B, P, 2]
trans          arange reshaped to [B, P, 3]
rotate         identity [B, P, 3, 3]
exist_logit    2.0 [B, P, 1]
exist          0.25 [B, P, 1]
rotation_quat  ones [B, P, 4]
```

It intentionally contains both `exist_logit` and `exist` so tests can verify
that packing prefers logits.

## Tests

### `test_pack_decoder_primitive_features_uses_matrix_rotation_and_logit`

Calls:

```python
pack_decoder_primitive_features(_outdict())
```

Checks:

```text
packed shape [2, 3, 18]
last channel == 2.0
```

Purpose:

Verify decoder primitive code layout:

```text
scale 3 + shape 2 + trans 3 + rotate matrix 9 + exist_logit 1 = 18
```

and verify `exist_logit` is used instead of `logit(exist)` when both are
present.

### `test_pack_serialized_primitive_features_uses_quaternion_when_available`

Calls:

```python
pack_serialized_primitive_features(_outdict(), rotation_mode="quat")
```

Checks:

```text
packed shape [2, 3, 13]
```

Purpose:

Verify compact serialization path:

```text
scale 3 + shape 2 + trans 3 + quaternion 4 + exist feature 1 = 13
```

This path is for reporting/storage, not decoder conditioning.

### `test_repeat_by_part_ids_repeats_slots_per_surface_point`

Creates:

```text
values   [2, 3, 4]
part_ids [0, 2, 1, 2]
```

Calls:

```python
repeat_by_part_ids(values, part_ids)
```

Checks:

```text
repeated shape [2, 4, 4]
repeated[:, 0] == values[:, 0]
repeated[:, 1] == values[:, 2]
repeated[:, 2] == values[:, 1]
```

Purpose:

Protect the decoder's per-surface-point feature construction. Every sampled
point must receive the primitive features and residual of its parent primitive.
