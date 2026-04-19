# models

This folder contains AutoDec neural submodules. These modules are intentionally
small and inspectable. The encoder and decoder compose them from the package
root.

Files:

```text
__init__.py
heads.py
offset_decoder.py
residual.py
```

## `__init__.py`

Exports:

```python
from autodec.models.heads import SuperDecHead
from autodec.models.offset_decoder import CrossAttentionOffsetDecoder
from autodec.models.residual import PartResidualProjector
```

The public names are:

```text
SuperDecHead
PartResidualProjector
CrossAttentionOffsetDecoder
```

This file has no model logic. It is an import convenience layer.

## `heads.py`

Defines:

```text
SuperDecHead
```

This is the AutoDec-owned version of the SuperDec primitive prediction head.
The main AutoDec-specific change is that it exposes raw existence logits:

```text
exist_logit [B, P, 1]
exist       [B, P, 1] = sigmoid(exist_logit)
```

### Input

```text
x [B, P, H]
```

where `x` is the final primitive query feature tensor from
`AutoDecEncoder`.

### Learned heads

The base head has these linear layers:

```text
scale_head Linear(H -> 3)
shape_head Linear(H -> 2)
rot_head   Linear(H -> 4) by default, or Linear(H -> 6) if rotation6d=True
t_head     Linear(H -> 3)
exist_head Linear(H -> 1)
```

If `ctx.extended=True`, the file also creates:

```text
tapering_head  Linear(H -> 2)
bending_k_head Linear(H -> 3)
bending_a_head Linear(H -> 3)
```

The extended fields are passed through to the output dictionary but are not used
by the current decoder or losses.

### Scale activation

Code:

```python
scale = torch.sigmoid(self.scale_head(x))
```

Shape:

```text
scale [B, P, 3]
```

Range:

```text
0 < scale < 1
```

### Shape activation

Code:

```python
shape = 0.1 + 1.8 * torch.sigmoid(self.shape_head(x))
```

Shape:

```text
shape [B, P, 2]
```

Range:

```text
0.1 < shape < 1.9
```

The two channels are the superquadric exponents:

```text
epsilon_1, epsilon_2
```

### Rotation path

If `ctx.rotation6d` is false, which is the default:

```text
rot_raw        [B, P, 4]
rotation_quat  [B, P, 4] = normalize(rot_raw)
rotate         [B, P, 3, 3] = quat2mat(rotation_quat)
```

The quaternion convention in this file is:

```text
q = [w, x, y, z]
```

The matrix is built from the standard normalized-quaternion formula:

```text
R00 = w^2 + x^2 - y^2 - z^2
R01 = 2xy - 2wz
R02 = 2wy + 2xz
R10 = 2wz + 2xy
R11 = w^2 - x^2 + y^2 - z^2
R12 = 2yz - 2wx
R20 = 2xz - 2wy
R21 = 2wx + 2yz
R22 = w^2 - x^2 - y^2 + z^2
```

If `ctx.rotation6d` is true:

```text
rot_raw     [B, P, 6]
rotation_6d [B, P, 6]
rotate      [B, P, 3, 3] = rot6d2mat(rot_raw)
```

`rot6d2mat` orthonormalizes two 3D vectors:

```text
a1 = rot6d[..., :3]
a2 = rot6d[..., 3:]
b1 = normalize(a1)
b2 = normalize(a2 - dot(b1, a2) * b1)
b3 = cross(b1, b2)
R = [b1, b2, b3]
```

### Translation path

Code:

```python
translation = self.t_head(x)
```

Shape:

```text
trans [B, P, 3]
```

There is no activation on translation.

### Existence path

Code:

```python
exist_logit = self.exist_head(x)
exist = torch.sigmoid(exist_logit)
```

Shapes:

```text
exist_logit [B, P, 1]
exist       [B, P, 1]
```

The decoder uses `exist_logit` for soft gates:

```text
w = sigmoid(exist_logit / tau)
```

The losses use `exist_logit` for BCE-with-logits when available.

### Output dictionary

Always returned:

```text
scale        [B, P, 3]
shape        [B, P, 2]
rotate       [B, P, 3, 3]
trans        [B, P, 3]
exist_logit  [B, P, 1]
exist        [B, P, 1]
```

Rotation-specific:

```text
rotation_quat [B, P, 4], when rotation6d=False
rotation_6d   [B, P, 6], when rotation6d=True
```

Optional extended output:

```text
tapering  [B, P, 2]
bending_k [B, P, 3]
bending_a [B, P, 3]
```

## `residual.py`

Defines:

```text
PartResidualProjector
```

Purpose:

Build one residual latent per primitive by combining the primitive query feature
with assignment-weighted local point evidence.

Constructor:

```python
PartResidualProjector(feature_dim=128, residual_dim=64, hidden_dim=None, eps=1e-6)
```

Stored dimensions:

```text
feature_dim   H
residual_dim  D
hidden_dim    H by default
eps           1e-6 by default
```

Network:

```text
Linear(2H -> hidden_dim)
ReLU
Linear(hidden_dim -> D)
```

### `pool_point_features`

Signature:

```python
pool_point_features(point_features, assign_matrix)
```

Inputs:

```text
point_features [B, N, H]
assign_matrix  [B, N, P]
```

Assignment mass:

```text
mass[b, j] = sum_i assign_matrix[b, i, j]
```

The implementation clamps mass:

```text
mass = clamp_min(mass, eps)
```

Pooled feature:

```text
pooled[b, j, h] =
  sum_i assign_matrix[b, i, j] * point_features[b, i, h]
  / mass[b, j]
```

Code:

```python
pooled = torch.einsum("bnp,bnh->bph", assign_matrix, point_features)
pooled = pooled / mass.unsqueeze(-1)
```

Output:

```text
pooled [B, P, H]
```

### `forward`

Signature:

```python
forward(sq_features, point_features, assign_matrix, return_pooled=False)
```

Inputs:

```text
sq_features    [B, P, H]
point_features [B, N, H]
assign_matrix  [B, N, P]
```

Computation:

```text
pooled = pool_point_features(point_features, assign_matrix)
residual_input = concat(sq_features, pooled)  # [B, P, 2H]
residual = MLP(residual_input)                # [B, P, D]
```

Returns:

```text
residual [B, P, D]
```

If `return_pooled=True`, returns:

```text
(residual [B, P, D], pooled [B, P, H])
```

## `offset_decoder.py`

Defines:

```text
CrossAttentionOffsetDecoder
build_offset_decoder
```

Purpose:

Predict a 3D offset for every sampled superquadric surface point. This is the
Option B decoder core.

### `CrossAttentionOffsetDecoder`

Constructor:

```python
CrossAttentionOffsetDecoder(
    point_in_dim,
    primitive_in_dim,
    hidden_dim=128,
    n_heads=4,
    offset_scale=None,
    n_blocks=1,
    self_attention_mode="none",
)
```

Submodules:

```text
point_proj       Linear(point_in_dim -> hidden_dim)
primitive_proj   Linear(primitive_in_dim -> hidden_dim)
blocks           n_blocks x OffsetDecoderBlock
offset_mlp       Linear(hidden_dim -> hidden_dim), ReLU, Linear(hidden_dim -> 3)
```

Each `OffsetDecoderBlock` applies optional point self-attention, primitive-token
cross-attention, and an FFN. `self_attention_mode="within_primitive"` groups
the primitive-major sampled surface points as `[B*P, S, H]` so self-attention is
local to each primitive instead of full `[P*S, P*S]` attention.

Inputs to `forward`:

```text
point_features   [B, M, point_in_dim]
primitive_tokens [B, P, primitive_in_dim]
```

Projection:

```text
point_hidden     = point_proj(point_features)      [B, M, H_dec]
primitive_hidden = primitive_proj(primitive_tokens) [B, P, H_dec]
```

Attention:

```text
query = point_hidden
key   = primitive_hidden
value = primitive_hidden
```

PyTorch's `nn.MultiheadAttention` applies its own internal Q/K/V projections
after these pre-projections. Reusing `primitive_proj` for key and value inputs
is therefore not equivalent to sharing the final attention K and V matrices.

Attention output:

```text
attended          [B, M, H_dec]
attention_weights [B, M, P], from the last block when return_attention=True
```

Offset MLP input:

```text
final block output
```

Offset output:

```text
offsets [B, M, 3]
```

If `offset_scale` is set:

```text
offsets = offset_scale * tanh(offsets)
```

This optional bound is disabled by default.

### `build_offset_decoder`

Factory function:

```python
build_offset_decoder(
    decoder_type,
    point_in_dim,
    primitive_in_dim,
    hidden_dim=128,
    n_heads=4,
    offset_scale=None,
    n_blocks=1,
    self_attention_mode="none",
)
```

Currently supported:

```text
decoder_type == "cross_attention"
```

Returns:

```text
CrossAttentionOffsetDecoder
```

Any other decoder type raises:

```text
ValueError("Unsupported offset decoder type: ...")
```
