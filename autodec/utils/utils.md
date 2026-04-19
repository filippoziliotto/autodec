# utils

This folder contains small tensor utilities used by the AutoDec decoder and
serialization/reporting code.

Files:

```text
__init__.py
checkpoints.py
inference.py
metrics.py
packing.py
```

## `__init__.py`

Exports:

```python
from autodec.utils.packing import (
    pack_decoder_primitive_features,
    pack_serialized_primitive_features,
    repeat_by_part_ids,
)
```

Public names:

```text
active_decoded_point_count
active_primitive_count
offset_ratio
pack_decoder_primitive_features
pack_serialized_primitive_features
primitive_mass_entropy
repeat_by_part_ids
scaffold_vs_decoded_chamfer
```

## `checkpoints.py`

Defines:

```text
strip_module_prefix
extract_state_dict
load_superdec_encoder_checkpoint
load_autodec_checkpoint
save_autodec_checkpoint
```

Purpose:

Checkpoint handling for the two different training needs:

1. initialize `AutoDec.encoder` from an old SuperDec checkpoint;
2. resume/save full AutoDec checkpoints.

### `strip_module_prefix`

Removes a leading DDP prefix:

```text
module.
```

from every state-dict key.

### `extract_state_dict`

Accepts either:

```text
checkpoint["model_state_dict"]
```

or a raw state dict.

Returns the model state dict.

### `load_superdec_encoder_checkpoint`

Signature:

```python
load_superdec_encoder_checkpoint(model_or_encoder, path, map_location="cpu", strict=True)
```

Behavior:

- loads a checkpoint with `torch.load(..., weights_only=False)`;
- extracts and strips the state dict;
- if `model_or_encoder` has `.encoder`, loads into that;
- otherwise loads directly into `model_or_encoder`.

This is used for phase-1 initialization from pretrained SuperDec weights.

### `load_autodec_checkpoint`

Signature:

```python
load_autodec_checkpoint(
    model,
    path,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    load_optimizer=True,
)
```

Behavior:

- loads full AutoDec model weights;
- optionally restores optimizer and scheduler states;
- returns metadata:

  ```text
  epoch
  val_loss
  ```

### `save_autodec_checkpoint`

Saves:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
epoch
val_loss
```

Handles DDP-wrapped models by saving `model.module.state_dict()`.

## `inference.py`

Defines:

```text
prune_decoded_points
```

Purpose:

Inference/evaluation post-processing for fixed-size decoded point clouds. It
uses primitive existence probabilities and `part_ids` to keep decoded points
from active primitives only. If `target_count` is provided, it deterministically
downsamples or repeats the pruned points so metrics can still consume a dense
`[B, target_count, 3]` tensor.

## `metrics.py`

Defines:

```text
offset_ratio
active_primitive_count
active_decoded_point_count
primitive_mass_entropy
scaffold_vs_decoded_chamfer
```

### `offset_ratio`

Computes:

```text
mean ||decoded_offsets||_2 / mean ||surface_points||_2
```

This monitors whether the learned residual offsets are dominating the SQ
scaffold.

### `active_primitive_count`

Input:

```text
exist [B, P] or [B, P, 1]
```

Output:

```text
mean_b sum_j 1[exist[b,j] > threshold]
```

Default threshold:

```text
0.5
```

### `active_decoded_point_count`

Input:

```text
decoded_weights [B, M] or [B, M, 1]
```

Output:

```text
mean_b sum_m 1[decoded_weights[b,m] > threshold]
```

### `primitive_mass_entropy`

Input:

```text
assign_matrix [B, N, P]
```

Mass:

```text
mass[b,j] = mean_i assign_matrix[b,i,j]
```

Entropy:

```text
-sum_j p[b,j] log(p[b,j])
```

where `p` is normalized primitive mass.

### `scaffold_vs_decoded_chamfer`

Computes two weighted Chamfer values:

```text
scaffold_chamfer = Chamfer(surface_points, target_points)
decoded_chamfer  = Chamfer(decoded_points, target_points)
```

Returns detached Python floats in a dictionary.

## `packing.py`

Defines:

```text
pack_decoder_primitive_features
pack_serialized_primitive_features
repeat_by_part_ids
```

It also contains two private helpers:

```text
_exist_feature
_matrix_to_quaternion
```

## `_exist_feature`

Purpose:

Return a scalar existence feature per primitive. The decoder wants the raw
logit, not the probability, when possible.

Input:

```text
outdict
```

Behavior:

```text
if "exist_logit" in outdict:
    return outdict["exist_logit"]
else:
    exist = clamp(outdict["exist"], 1e-6, 1 - 1e-6)
    return logit(exist)
```

Expected shape:

```text
[B, P, 1]
```

The fallback is for compatibility with SuperDec heads that expose only
`exist`.

## `pack_decoder_primitive_features`

Purpose:

Build the primitive feature tensor consumed by `AutoDecDecoder`.

Input keys:

```text
scale        [B, P, 3]
shape        [B, P, 2]
trans        [B, P, 3]
rotate       [B, P, 3, 3]
exist_logit  [B, P, 1], preferred
exist        [B, P, 1], fallback
```

Computation:

```text
rotate_flat = rotate.reshape(B, P, 9)
E_dec = concat(scale, shape, trans, rotate_flat, exist_feature)
```

Shape:

```text
E_dec [B, P, 18]
```

Channel layout:

```text
0:3    scale
3:5    shape
5:8    translation
8:17   flattened 3x3 rotation matrix
17:18  existence logit or reconstructed logit
```

Why 18:

The project notation describes a compact primitive code with fewer rotation
channels. This implementation passes the rotation matrix already produced by
the SuperDec head. That keeps the decoder independent of whether the head used
quaternion or 6D rotation internally.

## `_matrix_to_quaternion`

Purpose:

Convert a rotation matrix to a normalized quaternion when a serialized
quaternion code is requested but the outdict does not already contain
`rotation_quat`.

Input:

```text
rotation [*, 3, 3]
```

Output:

```text
quat [*, 4]
```

Convention:

```text
[w, x, y, z]
```

Implementation:

- Flattens batch dimensions.
- Uses the positive-trace formula when matrix trace is positive.
- Uses the largest diagonal entry fallback otherwise.
- Normalizes the final quaternion.

This helper is for reporting/storage, not for decoder conditioning.

## `pack_serialized_primitive_features`

Purpose:

Build a compact primitive code for reporting or storage. This is separate from
`E_dec`.

Signature:

```python
pack_serialized_primitive_features(outdict, rotation_mode="quat")
```

Common input keys:

```text
scale [B, P, 3]
shape [B, P, 2]
trans [B, P, 3]
exist_logit or exist
```

### `rotation_mode="quat"`

Uses:

```text
outdict["rotation_quat"] [B, P, 4]
```

if present. Otherwise computes quaternion from:

```text
outdict["rotate"] [B, P, 3, 3]
```

Output layout:

```text
scale 3 + shape 2 + trans 3 + quaternion 4 + exist_feature 1
```

Shape:

```text
[B, P, 13]
```

### `rotation_mode="6d"`

Uses:

```text
outdict["rotation_6d"] [B, P, 6]
```

if present. Otherwise uses the first two columns of the rotation matrix and
reshapes them to six values.

Output layout:

```text
scale 3 + shape 2 + trans 3 + rotation_6d 6 + exist_feature 1
```

Shape:

```text
[B, P, 15]
```

### Unsupported rotation modes

Any other value raises:

```text
ValueError("Unsupported rotation_mode: ...")
```

## `repeat_by_part_ids`

Purpose:

Repeat a per-primitive tensor so every sampled surface point receives the tensor
of its parent primitive.

Signature:

```python
repeat_by_part_ids(values, part_ids)
```

Inputs:

```text
values   [B, P, C]
part_ids [M]
```

Output:

```text
values[:, part_ids, :] [B, M, C]
```

Typical usage in the decoder:

```text
primitive_features [B, P, 18] -> [B, P*S, 18]
residual           [B, P, D]  -> [B, P*S, D]
```

`part_ids` comes from `SQSurfaceSampler` and is primitive-major.
