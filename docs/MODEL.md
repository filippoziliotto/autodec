# AutoDec Model Data Flow

This document describes the current `autodec/` implementation as code, not as
an idealized project proposal. It is meant to be used as a correctness checklist
for the model, decoder, and losses that currently exist in this repository.

The implemented package is modular:

- `autodec.encoder.AutoDecEncoder` produces SuperDec-style primitive parameters
  plus per-primitive residual latents.
- `autodec.decoder.AutoDecDecoder` turns primitive parameters and residuals into
  decoded point coordinates.
- `autodec.autodec.AutoDec` composes the encoder and decoder behind a single
  model call.
- `autodec.losses.AutoDecLoss` computes reconstruction and optional primitive
  regularization losses.

The implemented end-to-end data flow is:

```python
model = AutoDec(ctx)
loss_fn = AutoDecLoss(...)

outdict = model(points)
loss, metrics = loss_fn(batch, outdict)
```

All shapes below use these symbols:

```text
B   batch size
N   number of input points
P   number of predicted superquadric primitive slots
H   feature dimension from the SuperDec/PVCNN encoder
D   residual latent dimension per primitive
S   number of surface samples per primitive in the decoder
M   total decoded/scaffold points = P * S
L   number of SuperDec transformer decoder layers
```

In the default design, `P` is the configured `ctx.decoder.n_queries`, `H` is
`ctx.point_encoder.l3.out_channels`, `D` defaults to `64`, and `S` defaults to
`256`.

## Package Exports

The root package `autodec/__init__.py` exports:

```text
AutoDec
AutoDecEncoder
AutoDecDecoder
AutoDecLoss
PartResidualProjector
```

The loss package `autodec/losses/__init__.py` exports:

```text
AutoDecLoss
SQRegularizer
assignment_parsimony_loss
existence_loss
weighted_chamfer_l2
```

## Full Pipeline Summary

The implemented forward path is:

```text
points [B, N, 3]
  -> AutoDec
  -> AutoDecEncoder
     -> point_features [B, N, H]
     -> sq_features [B, P, H]
     -> assign_matrix [B, N, P]
     -> primitive parameters:
        scale [B, P, 3]
        shape [B, P, 2]
        rotate [B, P, 3, 3]
        trans [B, P, 3]
        exist_logit [B, P, 1]
        exist [B, P, 1]
     -> residual [B, P, D]
  -> AutoDecDecoder
     -> sampled SQ surface points [B, M, 3]
     -> decoded weights [B, M]
     -> Fourier position features [B, M, 3 + 6F]
     -> split projected decoder point features [B, M, 4C]
     -> split projected primitive attention tokens [B, P, 2C]
     -> decoded offsets [B, M, 3]
     -> decoded points [B, M, 3]
  -> AutoDecLoss
     -> reconstruction loss
     -> optional SQ fitting, parsimony, existence, and consistency terms
```

`AutoDec.forward(points)` is the normal training and evaluation entry point.
The lower-level `AutoDecEncoder` and `AutoDecDecoder` modules can still be used
directly for ablations or tests.

The explicit primitive code used by the neural decoder is not the 12-value
project-spec notation. The code uses the actual SuperDec rotation matrix:

```text
E_dec_j = [
  scale_j          3 values
  shape_j          2 values
  trans_j          3 values
  rotate_j         9 values, flattened 3x3 matrix
  exist_logit_j    1 value
]

E_dec_j has 18 values.
```

With `component_feature_dim=0`, the legacy raw decoder path has:

```text
3 + 18 + 64 + 1 = 86 channels
```

and each primitive attention token has:

```text
18 + 64 = 82 channels
```

The default path uses split projections instead. With `hidden_dim=128`,
`component_feature_dim=None` resolves to `C=32`, so decoded surface-point
features have `4C=128` channels and primitive attention tokens have `2C=64`
channels.

## Encoder: `autodec/encoder.py`

### Purpose

`AutoDecEncoder` is a SuperDec-compatible encoder with one new output branch:
`residual`, a per-primitive latent vector. It reuses SuperDec's point encoder and
transformer decoder, and owns an AutoDec-specific head that exposes raw
existence logits.

### Constructor Inputs

```python
AutoDecEncoder(ctx, point_encoder=None, layers=None, heads=None)
```

The `ctx` object is expected to contain at least:

```text
ctx.decoder.n_layers
ctx.decoder.n_heads
ctx.decoder.n_queries
ctx.decoder.deep_supervision
ctx.decoder.pos_encoding_type
ctx.decoder.dim_feedforward
ctx.decoder.swapped_attention
ctx.decoder.masked_attention
ctx.point_encoder.l3.out_channels
ctx.point_encoder.{l1,l2,l3}
```

Optional values used by the implementation:

```text
ctx.residual_dim, default 64
ctx.head_type, default "heads"
ctx.clear_orientation_heads, default False
ctx.rotation6d, default False
ctx.extended, default False
ctx.extended_non_zero_init, default False
```

If `point_encoder`, `layers`, or `heads` are passed manually, those injected
modules are used. This is useful for tests and for replacing pieces while
keeping the same data contract.

### Encoder State

The encoder stores:

```text
self.n_layers          = ctx.decoder.n_layers
self.n_heads           = ctx.decoder.n_heads
self.n_queries         = ctx.decoder.n_queries
self.emb_dims          = ctx.point_encoder.l3.out_channels
self.residual_dim      = ctx.residual_dim or 64
self.init_queries      [P + 1, H], all zeros, registered buffer
self.point_encoder     SuperDec StackedPVConv unless injected
self.layers            SuperDec TransformerDecoder unless injected
self.heads             AutoDec/SuperDec head unless injected
self.residual_projector PartResidualProjector(H, D)
```

The buffer `init_queries` has `P + 1` query tokens. AutoDec, like SuperDec,
uses the first `P` tokens as primitive tokens. The last token is kept in the
transformer sequence, but is removed before primitive parameter prediction and
assignment output.

### Input

`AutoDecEncoder.forward(x, return_features=True)` expects:

```text
x: [B, N, 3]
```

where each point has XYZ coordinates.

### Step 1: Point Feature Extraction

Code:

```python
point_features = self.point_encoder(x)
```

Output:

```text
point_features: [B, N, H]
```

This calls `superdec.models.point_encoder.StackedPVConv` unless a custom point
encoder was injected.

#### SuperDec Dependency: `superdec/models/point_encoder.py`

`StackedPVConv` is a sequence of three `PVConv` blocks:

```text
PVConv(ctx.l1) -> PVConv(ctx.l2) -> PVConv(ctx.l3)
```

Its input is `[B, N, 3]`. Internally it transposes to `[B, 3, N]` and uses the
coordinates as both initial point features and coordinates:

```python
coords_transpose = coords.transpose(-1, -2)  # [B, 3, N]
outputs_transpose, _ = self.stacked_pvconv((coords_transpose, coords_transpose))
return outputs_transpose.transpose(-1, -2)   # [B, N, H]
```

Each `PVConv` receives:

```text
features: [B, C_in, N]
coords:   [B, 3, N]
```

Each `PVConv` has two branches:

1. Voxel branch.
2. Point branch.

The voxel branch:

```text
coords are detached
coords are centered by subtracting their pointwise mean
coords are normalized by max norm if ctx.voxelization.normalize is true
coords are mapped into a resolution-R voxel grid
features are average-voxelized into [B, C_in, R, R, R]
Conv3d(C_in -> C_out, kernel_size)
GroupNorm(8, C_out)
Swish
Conv3d(C_out -> C_out, kernel_size)
GroupNorm(8, C_out)
trilinear devoxelization back to [B, C_out, N]
```

The point branch is `SharedMLP`:

```text
Conv1d(C_in -> C_out, kernel_size=1)
GroupNorm(8, C_out)
Swish
```

The two branches are added:

```text
fused_features = voxel_features + point_features(features)
```

So, after the three PVConv blocks, AutoDec receives dense point features:

```text
F_PC = point_features in R^{B x N x H}
```

### Step 2: SuperDec Transformer Query Decoder

Code:

```python
refined_queries_list, assign_matrices = self.layers(self.init_queries, point_features)
```

Inputs:

```text
self.init_queries: [P + 1, H]
point_features:    [B, N, H]
```

Outputs:

```text
refined_queries_list: list length L, each [B, P + 1, H]
assign_matrices:      list length L, each [B, N, P]
```

The values in `assign_matrices` at this point are assignment logits, not yet
soft assignments.

#### SuperDec Dependency: `superdec/models/decoder.py`

`TransformerDecoder` stores `L` decoder layers and a positional encoding. The
positional encoding is either:

- sinusoidal, `SinusoidalPositionalEncoding`, or
- learned, `LearnablePositionalEncoding`.

The query input is repeated across the batch and positional encoding is added:

```text
Q_0 = positional_encoding(init_queries repeated B times)
Q_0: [B, P + 1, H]
```

For each layer `l`, the transformer computes:

```text
Q_l = DecoderLayer(Q_{l-1}, memory=F_PC, memory_mask=mask)
```

where `F_PC` is the point feature memory `[B, N, H]`.

Then assignment logits are computed by a query projection:

```python
projected_queries_layer = self.project_queries(output)
assign_matrix = memory @ projected_queries_layer.transpose(-1, -2)
assign_matrix = assign_matrix[..., :-1]
```

Shape detail:

```text
memory:                  [B, N, H]
projected_queries_layer: [B, P + 1, H]
transpose:               [B, H, P + 1]
raw product:             [B, N, P + 1]
after dropping last:     [B, N, P]
```

The last query token is not assigned as a primitive. It is discarded by
`[..., :-1]`.

If `ctx.decoder.masked_attention` is enabled, the SuperDec transformer builds a
mask from the softmax of the assignment logits:

```text
soft_assign = softmax(assign_logits, dim=-1)
mask = soft_assign > 0.5
```

The mask is transposed to primitive-major form and an all-true row is appended
for the extra query token. This mask is then passed as `memory_mask` to the next
decoder layer. AutoDec does not change this SuperDec behavior.

#### SuperDec Dependency: `superdec/models/decoder_layer.py`

`DecoderLayer` subclasses PyTorch `TransformerDecoderLayer`. It supports the
normal decoder order:

```text
self-attention over query tokens
cross-attention from query tokens to point memory
feed-forward block
```

or a swapped order if `swapped_attention=True`:

```text
cross-attention first
self-attention second
feed-forward block
```

Both `norm_first=True` and `norm_first=False` paths are inherited from PyTorch
and supported by the subclass. The repository configs normally build the layer
with `batch_first=True`, so tensors are `[B, sequence, H]`.

### Step 3: Primitive Head Per Transformer Layer

Code:

```python
for query_features, assign_logits in zip(refined_queries_list, assign_matrices):
    sq_features = query_features[:, :-1, ...]
    outdict = self.heads(sq_features)
    outdict = self._ensure_exist_logit(outdict)
    outdict["assign_matrix"] = torch.softmax(assign_logits, dim=2)
    outdict_list.append(outdict)
```

For every transformer layer, AutoDec can build a primitive output dictionary.
Only the last one is returned:

```python
outdict = outdict_list[-1]
sq_features = refined_queries_list[-1][:, :-1, ...]
```

Shapes:

```text
query_features from layer l: [B, P + 1, H]
sq_features:                 [B, P, H]
assign_logits:               [B, N, P]
assign_matrix:               [B, N, P]
```

The soft assignment matrix is:

```text
M_{b,i,j} = exp(A_{b,i,j}) / sum_{k=1..P} exp(A_{b,i,k})
```

where `A` is the raw assignment logit tensor. Because the softmax is along
`dim=2`, every input point distributes its mass across primitive slots:

```text
sum_j M_{b,i,j} = 1
```

### Step 4: AutoDec Primitive Head

If `ctx.head_type == "heads"` or absent, AutoDec uses
`autodec.models.heads.SuperDecHead`. This is a copied and modified version of
`superdec.models.heads.SuperDecHead`. The important AutoDec modification is
that it returns both:

```text
exist_logit [B, P, 1]
exist       [B, P, 1]
```

The original `superdec.models.heads.SuperDecHead` returns only `exist`.

The AutoDec head receives:

```text
sq_features [B, P, H]
```

and computes:

```text
scale_pre = Linear(H -> 3)(sq_features)
scale     = sigmoid(scale_pre)
scale     [B, P, 3]
```

So each scale component is in `(0, 1)`.

```text
shape_pre = Linear(H -> 2)(sq_features)
shape     = 0.1 + 1.8 * sigmoid(shape_pre)
shape     [B, P, 2]
```

So each shape exponent is in `(0.1, 1.9)`.

```text
rot_raw = Linear(H -> 4)(sq_features) if rotation6d is false
q       = normalize(rot_raw, p=2)
rotate  = quat2mat(q)
rotate  [B, P, 3, 3]
```

or:

```text
rot_raw = Linear(H -> 6)(sq_features) if rotation6d is true
rotate  = rot6d2mat(rot_raw)
rotate  [B, P, 3, 3]
```

For quaternion rotation, `quat2mat` uses normalized quaternions:

```text
q = [w, x, y, z]
```

and expands the standard quaternion-to-matrix expression:

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

For 6D rotation, `rot6d2mat` follows the Zhou-style orthonormalization:

```text
a1 = rot6d[..., 0:3]
a2 = rot6d[..., 3:6]
b1 = normalize(a1)
b2 = normalize(a2 - dot(b1, a2) * b1)
b3 = cross(b1, b2)
R  = [b1, b2, b3] as matrix columns
```

Translation is unbounded:

```text
trans = Linear(H -> 3)(sq_features)
trans [B, P, 3]
```

Existence is:

```text
exist_logit = Linear(H -> 1)(sq_features)
exist       = sigmoid(exist_logit)
```

The head returns at least:

```text
scale        [B, P, 3]
shape        [B, P, 2]
rotate       [B, P, 3, 3]
trans        [B, P, 3]
exist_logit  [B, P, 1]
exist        [B, P, 1]
```

If quaternion rotation is used, it also returns:

```text
rotation_quat [B, P, 4]
```

If 6D rotation is used, it also returns:

```text
rotation_6d [B, P, 6]
```

If `ctx.extended=True`, it also predicts SuperDec's extended deformation
fields:

```text
tapering  [B, P, 2] = tanh(tapering_head(sq_features))
bending_k [B, P, 3] = (sigmoid(raw) * 0.95) / softmax-like max scale
bending_a [B, P, 3] = sigmoid(raw) * 2*pi
```

AutoDec's decoder and losses currently do not consume these extended fields.
They remain in `outdict` if the head emits them.

### Alternate Head Types

`AutoDecEncoder._build_heads` supports:

```text
head_type == "heads"      -> autodec.models.heads.SuperDecHead
head_type == "heads_mlp"  -> superdec.models.heads_mlp.SuperDecHead
head_type == "heads_mlps" -> superdec.models.heads_mlps.SuperDecHead
```

Only the AutoDec `"heads"` variant natively returns `exist_logit`. The
SuperDec MLP variants return only `exist`. To keep the downstream decoder
contract valid, `AutoDecEncoder._ensure_exist_logit` reconstructs the logit:

```text
exist_clamped = clamp(exist, 1e-6, 1 - 1e-6)
exist_logit   = logit(exist_clamped)
```

This fallback is numerically clipped because exact `0` or `1` probabilities
would have infinite logits.

### Step 5: Optional LM Optimization

The encoder has:

```python
self.lm_optimization = False
```

So the LM path is currently disabled. If it were enabled, the encoder would
lazy-import:

```python
from superdec.lm_optimization.lm_optimizer import LMOptimizer
```

and apply:

```python
outdict = self.lm_optimizer(outdict, x)
```

after primitive prediction and before residual projection. In the current code,
this never runs.

### Step 6: Per-Primitive Residual Projection

Code:

```python
residual, pooled = self.residual_projector(
    sq_features,
    point_features,
    outdict["assign_matrix"],
    return_pooled=True,
)
outdict["residual"] = residual
outdict["pooled_features"] = pooled
```

Inputs:

```text
sq_features:    [B, P, H]
point_features: [B, N, H]
assign_matrix:  [B, N, P]
```

Module: `autodec.models.residual.PartResidualProjector`.

The projector first computes assignment mass per primitive:

```text
mass_{b,j} = sum_i M_{b,i,j}
mass       [B, P]
```

with clamp:

```text
mass = clamp_min(mass, eps)
eps  = 1e-6 by default
```

Then it pools point features into each primitive. The mean-only diagnostic
helper is:

```text
g_{b,j,h} = sum_i M_{b,i,j} * F_PC_{b,i,h} / mass_{b,j}
```

Code:

```python
pooled = torch.einsum("bnp,bnh->bph", assign_matrix, point_features)
pooled = pooled / mass.unsqueeze(-1)
```

Shape:

```text
pooled_features = g_mean [B, P, H]
```

The residual input is:

```text
g_max,j = max_i masked_by(M_{b,i,j} > eps, M_{b,i,j} * F_PC_{b,i})
g_var,j = sum_i M_{b,i,j} * (F_PC_{b,i} - g_mean,j)^2 / mass_{b,j}

R_in_{b,j} = concat(F_SQ_{b,j}, g_mean,j, g_max,j, g_var,j)
R_in       [B, P, 4H]
```

Then a two-layer MLP maps it to the residual latent:

```text
Linear(4H -> hidden_dim)
ReLU
Linear(hidden_dim -> D)
```

where `hidden_dim` defaults to `H`.

Output:

```text
residual = Z [B, P, D]
```

This is the main AutoDec addition to the SuperDec encoder. It gives each
primitive a local neural residual that is conditioned on:

1. the primitive query state `F_SQ`, and
2. the mean, max, and variance of point evidence assigned to that primitive through `M`.

### Encoder Output Dictionary

With `return_features=True`, `AutoDecEncoder.forward` returns an `outdict` with
at least:

```text
scale             [B, P, 3]
shape             [B, P, 2]
rotate            [B, P, 3, 3]
trans             [B, P, 3]
exist_logit       [B, P, 1]
exist             [B, P, 1]
assign_matrix     [B, N, P]
residual          [B, P, D]
pooled_features   [B, P, H]
point_features    [B, N, H]
sq_features       [B, P, H]
```

Depending on configuration it may also contain:

```text
rotation_quat     [B, P, 4]
rotation_6d       [B, P, 6]
tapering          [B, P, 2]
bending_k         [B, P, 3]
bending_a         [B, P, 3]
```

## Decoder: `autodec/decoder.py`

### Purpose

`AutoDecDecoder` converts the explicit primitive bottleneck and residual latent
into a fixed-size decoded point cloud. It uses Option B from the project plan:
per-surface-point features plus cross-attention to primitive tokens.

### Constructor

```python
AutoDecDecoder(
    residual_dim=64,
    primitive_dim=18,
    n_surface_samples=256,
    hidden_dim=128,
    n_heads=4,
    exist_tau=1.0,
    angle_sampler=None,
    offset_scale=None,
    offset_cap=None,
    positional_frequencies=6,
    component_feature_dim=None,
    n_blocks=2,
    self_attention_mode="within_primitive",
)
```

Stored dimensions:

```text
D = residual_dim
primitive_dim = 18 by default
S = n_surface_samples
point_feature_dim = 3 + primitive_dim + D + 1
primitive_token_dim = primitive_dim + D
```

Those are the legacy dimensions when `component_feature_dim=0` and
`positional_frequencies=0`. The default path uses Fourier features and split
projections:

```text
position_feature_dim = 3 + 6 * positional_frequencies
component_feature_dim = max(4, hidden_dim // 4) if None
point_feature_dim = 4 * component_feature_dim
primitive_token_dim = 2 * component_feature_dim
```

For defaults:

```text
position_feature_dim = 39
component_feature_dim = 32
point_feature_dim = 128
primitive_token_dim = 64
```

Submodules:

```text
self.surface_sampler = SQSurfaceSampler(n_samples=S, tau=exist_tau)
self.offset_decoder  = CrossAttentionOffsetDecoder(...)
```

### Decoder Input

`AutoDecDecoder.forward(outdict, return_attention=False, return_consistency=False)` expects the encoder
dictionary to contain:

```text
scale        [B, P, 3]
shape        [B, P, 2]
rotate       [B, P, 3, 3]
trans        [B, P, 3]
exist_logit  [B, P, 1]
residual     [B, P, D]
```

It also preserves all existing keys by returning `result = dict(outdict)` plus
decoder keys.

## Superquadric Surface Sampling: `autodec/sampling/sq_surface.py`

### Purpose

`SQSurfaceSampler` creates differentiable surface points from predicted
superquadric parameters. It also computes differentiable soft existence weights
used by the decoder and losses.

### Angle Sampler

If `angle_sampler` is not injected, the sampler lazy-imports:

```python
from superdec.loss.sampler import EqualDistanceSamplerSQ
```

and constructs:

```python
EqualDistanceSamplerSQ(n_samples=S, D_eta=0.05, D_omega=0.05)
```

This is a SuperDec dependency. `EqualDistanceSamplerSQ.sample_on_batch` calls
the compiled/fast SuperDec sampler:

```python
fast_sample_on_batch(shapes, epsilons, n_samples)
```

Inputs to `sample_on_batch` are NumPy arrays:

```text
scale.detach().cpu().numpy()  [B, P, 3]
shape.detach().cpu().numpy()  [B, P, 2]
```

Before angle sampling and signed-power coordinate evaluation, the sampler
defensively clamps shape exponents to `[0.1, 2.0]`. Normal `SuperDecHead`
outputs already lie in `(0.1, 1.9)`, but this protects manually constructed or
externally loaded `outdict` values.

Therefore, gradients do not flow through the choice of sample angles. This is
intentional in the current code. Gradients still flow through the sampled point
coordinates with respect to `scale`, `shape`, `rotate`, and `trans`.

The returned angles are converted back to tensors on the scale tensor's device
and dtype:

```text
etas   [B, P, S]
omegas [B, P, S]
```

### Signed Power

The sampler uses:

```text
signed_power(v, e) = sign(v) * clamp(abs(v), eps)^e
```

with `eps=1e-6` by default.

Important behavior:

```text
sign(0) = 0
```

so exact zero trigonometric coordinates remain zero. The clamp avoids unstable
fractional powers near zero for nonzero signs.

### Canonical Superquadric Points

For each primitive:

```text
scale = [sx, sy, sz]
shape = [e1, e2]
eta   in [-pi/2, pi/2]
omega in [-pi, pi]
```

The implemented canonical parametric surface is:

```text
c_e(u) = sign(cos(u)) * max(abs(cos(u)), eps)^e
s_e(u) = sign(sin(u)) * max(abs(sin(u)), eps)^e

x = sx * c_e1(eta) * c_e2(omega)
y = sy * c_e1(eta) * s_e2(omega)
z = sz * s_e1(eta)
```

Shape:

```text
canonical_points [B, P, S, 3]
```

### World-Space Transform

The sampler then transforms from primitive coordinates to world coordinates:

```text
p_world = R_j * p_canonical + t_j
```

Code:

```python
surface = torch.matmul(rotate.unsqueeze(2), canonical.unsqueeze(-1)).squeeze(-1)
surface = surface + trans.unsqueeze(2)
```

Shapes:

```text
rotate.unsqueeze(2):    [B, P, 1, 3, 3]
canonical.unsqueeze(-1): [B, P, S, 3, 1]
surface:                [B, P, S, 3]
```

### Flattened Points, Part IDs, and Weights

The sampler returns an `SQSurfaceSample` dataclass with:

```text
canonical_points [B, P, S, 3]
surface_points   [B, P, S, 3]
flat_points      [B, P*S, 3]
part_ids         [P*S]
weights          [B, P*S]
```

The flattened order is primitive-major:

```text
part_ids = [0 repeated S times, 1 repeated S times, ..., P-1 repeated S times]
```

Existence gates are:

```text
w_{b,j} = sigmoid(exist_logit_{b,j} / tau)
```

where `tau` is `exist_tau` in the decoder or `sq_tau` in the SQ regularizer.
They are repeated for every sampled point from that primitive:

```text
weights_{b, j*S + s} = w_{b,j}
```

Shape:

```text
weights [B, P*S]
```

Critical implementation detail:

```text
weights do not multiply surface coordinates.
```

The scaffold point `surface_points` remains geometrically valid even for a low
existence weight. The gate is applied later to offsets and losses.

## Primitive Feature Packing: `autodec/utils/packing.py`

### Decoder Primitive Features

The decoder calls:

```python
primitive_features = pack_decoder_primitive_features(outdict)
```

The packed tensor is:

```text
E_dec = concat(scale, shape, trans, flatten(rotate), exist_feature)
```

where:

```text
scale           [B, P, 3]
shape           [B, P, 2]
trans           [B, P, 3]
flatten(rotate) [B, P, 9]
exist_feature   [B, P, 1]
```

The result is:

```text
E_dec [B, P, 18]
```

`exist_feature` is:

```text
exist_logit if present
logit(clamp(exist, 1e-6, 1 - 1e-6)) otherwise
```

The decoder uses the raw existence logit as a feature, not the sigmoid
probability.

### Serialized Primitive Features

`pack_serialized_primitive_features` is for reporting/storage, not for the
decoder forward path.

With `rotation_mode="quat"`:

```text
E_ser = concat(scale, shape, trans, rotation_quat, exist_feature)
E_ser shape = [B, P, 13]
```

If `rotation_quat` is absent, it reconstructs a quaternion from the rotation
matrix using `_matrix_to_quaternion`.

With `rotation_mode="6d"`:

```text
E_ser = concat(scale, shape, trans, rotation_6d, exist_feature)
E_ser shape = [B, P, 15]
```

If `rotation_6d` is absent, it uses the first two matrix columns:

```python
outdict["rotate"][..., :2].reshape(B, P, 6)
```

### Repeating Per-Primitive Values Per Surface Point

The decoder uses:

```python
repeat_by_part_ids(values, part_ids)
```

Code:

```python
return values[:, part_ids, :]
```

If:

```text
values   [B, P, C]
part_ids [M]
```

then:

```text
repeated [B, M, C]
```

This aligns each sampled surface point with the primitive code and residual
latent of the primitive it was sampled from.

## Decoder Feature Construction

Inside `AutoDecDecoder.forward`:

```python
sample = self.surface_sampler(outdict)
primitive_features = pack_decoder_primitive_features(outdict)
residual = outdict["residual"]
position_features = self.surface_position_features(sample.flat_points)

gates = sample.weights.unsqueeze(-1)
```

Shapes:

```text
sample.flat_points         [B, M, 3]
position_features          [B, M, 3 + 6F]
primitive_features         [B, P, 18]
residual                   [B, P, D]
gates                      [B, M, 1]
```

The default point-level decoder feature is built from separate projected
components:

```text
Proj_pos(position_features_i)
Proj_E(E_dec_{part(i)})
Proj_Z(Z_{part(i)})
Proj_gate(w_i)
```

Code:

```python
projected_position = self.position_projector(position_features)
projected_primitives = self.primitive_feature_projector(primitive_features)
projected_residual = self.residual_projector(residual)
projected_gates = self.gate_projector(gates)
decoder_features = cat(projected_position, projected_E_by_point, projected_Z_by_point, projected_gates)
```

Shape:

```text
decoder_features [B, M, 4C]
```

For the default `hidden_dim=128`:

```text
decoder_features [B, M, 128]
```

The primitive-token feature used for cross-attention is:

```text
T_j = concat(Proj_E(E_dec_j), Proj_Z(Z_j))
```

Code:

```python
primitive_tokens = torch.cat([projected_primitives, projected_residual], dim=-1)
```

Shape:

```text
primitive_tokens [B, P, 2C]
```

For the default `hidden_dim=128`:

```text
primitive_tokens [B, P, 64]
```

If `component_feature_dim=0`, the decoder disables the split projections and
uses the older raw concatenation:

```text
decoder_features = concat(position_features, E_dec_by_point, Z_by_point, gate)
primitive_tokens = concat(E_dec, Z)
```

## Offset Decoder: `autodec/models/offset_decoder.py`

### Purpose

`CrossAttentionOffsetDecoder` predicts a 3D offset for every sampled SQ surface
point. It is Option B from the project plan: an MLP-like point stream plus
cross-attention to primitive tokens.

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

Each block applies optional within-primitive self-attention, primitive
cross-attention, and an FFN, each with residual connection and LayerNorm.

### Offset Decoder Forward

Inputs:

```text
point_features    = decoder_features [B, M, point_in_dim]
primitive_tokens  [B, P, primitive_in_dim]
```

Projection:

```text
Q_points = point_proj(point_features)
K_prims  = primitive_proj(primitive_tokens)
V_prims  = primitive_proj(primitive_tokens)
```

Shapes:

```text
Q_points [B, M, hidden_dim]
K_prims  [B, P, hidden_dim]
V_prims  [B, P, hidden_dim]
```

For each block, when `self_attention_mode="within_primitive"`, sampled points
are reshaped to `[B * P, S, H]` and self-attended within their parent primitive.
Then the PyTorch cross-attention call is:

```python
attended, attention_weights = self.cross_attention(
    query=point_hidden,
    key=primitive_hidden,
    value=primitive_hidden,
    need_weights=True,
    average_attn_weights=True,
)
```

With `average_attn_weights=True`, attention weights are averaged across heads:

```text
attention_weights [B, M, P]
```

For each decoded surface point `i`, cross-attention computes a weighted sum of
primitive token values:

```text
a_{i,j} = softmax_j( Q_i K_j^T / sqrt(hidden_dim_per_head) )
attended_i = sum_j a_{i,j} V_j
```

The exact multi-head projections are inside `nn.MultiheadAttention`, but the
external contract is:

```text
attended [B, M, hidden_dim]
```

The residual offset head receives the final point representation after all
attention blocks:

```text
Delta_i_raw = offset_mlp(point_hidden_i)
```

Shape:

```text
decoded_offsets [B, M, 3]
```

If `offset_cap` is not `None`, `AutoDecDecoder` applies the preferred
primitive-scale bound after the offset decoder predicts raw offsets:

```text
Delta_i = tanh(Delta_i_raw) * offset_cap * mean(scale_{part(i)})
```

`offset_cap: 0.3` is enabled in the default AutoDec configs. Set
`offset_cap: null` to keep the older unbounded behavior. The cap is per
coordinate and scale-aware: points sampled from larger primitives can move
farther than points sampled from thin primitives.

The older scalar `offset_scale` path is still available for compatibility when
`offset_cap` is `None`:

```text
Delta_i = offset_scale * tanh(Delta_i_raw)
```

If both `offset_cap` and `offset_scale` are `None`:

```text
Delta_i = Delta_i_raw
```

### Decoded Points

Back in `AutoDecDecoder.forward`:

```python
decoded_points = sample.flat_points + gates * offsets
```

Formula:

```text
p_hat_{b,i} = p_sq_{b,i} + w_{b,i} * Delta_{b,i}
```

Shapes:

```text
p_sq           [B, M, 3]
w              [B, M, 1]
Delta          [B, M, 3]
p_hat          [B, M, 3]
```

This is the key AutoDec decoder rule:

```text
existence gates offset magnitude, not scaffold coordinates.
```

Low-existence primitives still produce scaffold points, but those points:

1. receive nearly zero learned displacement, and
2. are downweighted or penalized in the weighted losses.

Training intentionally keeps all `P*S` decoded points. This fixed-size tensor is
important for batching, differentiability, and stable loss computation. Hard
removal of inactive primitives belongs in inference or reporting code, not in
the training forward pass.

### Decoder Output Dictionary

`AutoDecDecoder.forward` returns a copy of the input `outdict` plus:

```text
surface_points            [B, M, 3]
surface_points_by_part    [B, P, S, 3]
canonical_surface_points  [B, P, S, 3]
decoded_weights           [B, M]
part_ids                  [M]
E_dec                     [B, P, 18]
decoder_features          [B, M, 3 + 18 + D + 1]
primitive_tokens          [B, P, 18 + D]
decoded_offsets           [B, M, 3]
decoded_points            [B, M, 3]
```

If `return_attention=True`, it also returns:

```text
decoder_attention         [B, M, P]
```

## Inference-Time Pruning

### Does Pruning Make Sense?

Yes, pruning makes sense for inference, visualization, and paper-style
evaluation. During training, inactive primitives are handled softly through
`decoded_weights` and the weighted Chamfer loss. At inference time, however,
leaving low-existence scaffold points in `decoded_points` can create a mismatch:
points from primitives that would be considered inactive still participate in
raw Chamfer metrics and still appear in point-cloud exports.

The right distinction is:

```text
training:    keep fixed [B, P*S, 3], use soft weights in the loss
inference:   drop or downselect points from inactive primitives
evaluation:  prune first, then optionally resample/repeat to a fixed point count
```

The standalone evaluator can now prune `outdict["decoded_points"]` before
paper-style metrics and test visualizations by setting
`eval.prune_decoded_points: true`. Training validation still uses the raw
fixed-size tensor and `AutoDecLoss`, which keeps validation aligned with the
training objective.

### Pruning Inputs

The model already emits everything needed for pruning:

```text
exist             [B, P, 1]       primitive existence probability
exist_logit       [B, P, 1]       raw existence logit
decoded_weights   [B, P*S]        sigmoid(exist_logit / tau), repeated per point
part_ids          [P*S]           primitive index for each decoded point
decoded_points    [B, P*S, 3]     decoded fixed-size point cloud
surface_points    [B, P*S, 3]     SQ scaffold point cloud
decoded_offsets   [B, P*S, 3]     predicted offsets
```

For pruning, prefer primitive-level `exist > threshold` when the goal is to
remove whole primitives. Use `decoded_weights > threshold` only if the gate
temperature is part of the inference policy.

### Implemented Post-Processing Helper

The pruning logic lives in `autodec/utils/inference.py`:

```python
def prune_decoded_points(
    outdict,
    exist_threshold=0.5,
    target_count=None,
):
    ...
```

Behavior:

1. Compute `primitive_active = outdict["exist"].squeeze(-1) > exist_threshold`.
2. Convert that to a point mask with `point_active = primitive_active[:, part_ids]`.
3. For each batch item, return `decoded_points[b, point_active[b]]`.
4. If no primitive is active, keep the highest-existence primitive rather than
   returning an empty point cloud.
5. If `target_count` is set, resample or repeat points to that count for fair
   fixed-cardinality metrics.

This keeps the model forward path unchanged and avoids mixing training behavior
with reporting behavior.

### Where To Add It

There are three useful integration levels:

```text
1. Visualization-only:
   For standalone test evaluation, this is implemented through
   autodec/eval/evaluator.py before calling AutoDecEpochVisualizer.
   Training visualizations are still raw unless a separate visualization-only
   config is added.

2. Evaluation metrics:
   Apply pruning in autodec/eval/evaluator.py before paper_chamfer_metrics.
   This is implemented and controlled by:
     eval.prune_decoded_points: true
     eval.prune_exist_threshold: 0.5
     eval.prune_target_count: null

3. Public inference API:
   The helper is public through autodec.utils.prune_decoded_points. A richer
   InferenceResult object could still return both:
     raw decoded_points    [B, P*S, 3]
     pruned point clouds   list[Tensor[N_b, 3]]
   This is best if downstream scripts will consume AutoDec predictions.
```

The implemented first step is option 2 plus test-evaluation visualizations and
the reusable helper. It intentionally does not change training validation loss
or checkpoint selection.

## Losses

The implemented losses live under `autodec/losses/`.

## Weighted Chamfer-L2: `autodec/losses/chamfer.py`

### Purpose

`weighted_chamfer_l2` compares the fixed-size decoded point cloud to the input
point cloud while using the decoder's soft existence weights.

Signature:

```python
weighted_chamfer_l2(
    pred,
    target,
    weights,
    eps=1e-6,
    min_backward_weight=1e-3,
    return_components=False,
)
```

Inputs:

```text
pred    [B, M, 3]       decoded points
target  [B, N, 3]       input/ground-truth points
weights [B, M] or [B, M, 1]
```

Input validation enforces:

```text
pred has rank 3 and last dim 3
target has rank 3 and last dim 3
weights has shape [B, M] after optional squeeze
pred and target have the same batch size
pred and weights match in [B, M]
```

Distances:

```text
D_{b,m,n} = ||pred_{b,m} - target_{b,n}||_2^2
```

Code:

```python
distances = torch.cdist(pred, target, p=2).pow(2)
```

Shape:

```text
distances [B, M, N]
```

### Forward Term

The forward term asks each predicted point to be close to some target point. It
is weighted by decoded existence.

Implementation:

```python
forward_weights = weights.clamp_min(eps)
forward = distances.min(dim=2).values
forward = (forward * forward_weights).sum(dim=1) / forward_weights.sum(dim=1)
```

Formula:

```text
d_fwd_{b,m} = min_n D_{b,m,n}

L_fwd_b =
  sum_m max(w_{b,m}, eps) * d_fwd_{b,m}
  / sum_m max(w_{b,m}, eps)

L_fwd = mean_b L_fwd_b
```

The clamp means exactly zero weights still contribute with tiny weight `eps`.
This avoids division by zero if all weights are zero.

### Backward Term

The backward term asks every target point to be covered by some predicted point,
but penalizes coverage by inactive decoded points.

Implementation:

```python
backward_weights = weights.clamp_min(min_backward_weight)
weighted_distances = distances / backward_weights.unsqueeze(-1)
backward = weighted_distances.min(dim=1).values.mean(dim=1)
```

Formula:

```text
D_weighted_{b,m,n} = D_{b,m,n} / max(w_{b,m}, min_backward_weight)

d_bwd_{b,n} = min_m D_weighted_{b,m,n}

L_bwd_b = (1/N) * sum_n d_bwd_{b,n}
L_bwd   = mean_b L_bwd_b
```

This discourages an inactive scaffold point from explaining an input point:
if `w` is tiny, its distance is divided by a small number before nearest
neighbor selection.

### Total Weighted Chamfer

```text
L_chamfer = L_fwd + L_bwd
```

If `return_components=True`, the function returns:

```text
(loss, {"forward": L_fwd, "backward": L_bwd})
```

Otherwise it returns only `loss`.

## SQ Regularizer: `autodec/losses/sq_regularizer.py`

### Purpose

`SQRegularizer` implements the v1 primitive fitting term. It is device-safe and
does not call `superdec.loss.loss.SuperDecLoss`, because the original SuperDec
loss moves batch tensors with `.cuda()` internally. AutoDec keeps the behavior
inside `autodec/` and avoids modifying `superdec/`.

The implemented regularizer is sampled-surface Chamfer-L2 without normal
alignment.

### Inputs

```python
regularizer(batch, outdict, return_components=False)
```

Required batch key:

```text
batch["points"] [B, N, 3]
```

Required `outdict` keys:

```text
scale         [B, P, 3]
shape         [B, P, 2]
rotate        [B, P, 3, 3]
trans         [B, P, 3]
assign_matrix [B, N, P]
exist or exist_logit
```

If `exist` is missing, the regularizer computes:

```text
exist = sigmoid(exist_logit)
```

If `exist_logit` is missing, the sampler helper reconstructs it from `exist`:

```text
exist_logit = logit(clamp(exist, 1e-6, 1 - 1e-6))
```

### Surface Sampling

`SQRegularizer` owns another `SQSurfaceSampler`:

```text
SQSurfaceSampler(n_samples=n_samples, tau=tau, angle_sampler=angle_sampler)
```

It samples:

```text
surface_points_by_part [B, P, S_sq, 3]
```

where `S_sq = n_samples` for this regularizer. This may be the same as the
decoder sample count or different, depending on how the loss is constructed.

The soft weights returned by the sampler are not used directly in the SQ
regularizer. The reverse SQ-to-point term uses `exist`.

### Distance Tensor

The implementation flattens the surface, computes distances, and reshapes:

```python
flat_surface = sample.surface_points.reshape(B, P * S_sq, 3)
distances = torch.cdist(flat_surface, points, p=2).pow(2)
distances = distances.view(B, P, S_sq, N)
```

So:

```text
D_{b,j,s,i} = ||x_sq_{b,j,s} - x_{b,i}||_2^2
D shape = [B, P, S_sq, N]
```

### Point-to-SQ Term

Implementation:

```python
point_to_sq = distances.min(dim=2).values.transpose(1, 2)
point_to_sq = (point_to_sq * assign_matrix).sum(dim=-1).mean(dim=-1)
point_to_sq = point_to_sq.mean()
```

Shape path:

```text
min over S_sq:       [B, P, N]
transpose to [B,N,P] [B, N, P]
multiply by M:       [B, N, P]
sum over P:          [B, N]
mean over N:         [B]
mean over B:         scalar
```

Formula:

```text
L_point_to_sq =
  mean_b (1/N) * sum_i sum_j M_{b,i,j} * min_s D_{b,j,s,i}
```

This term uses `assign_matrix` to decide which primitive should explain which
input points.

### SQ-to-Point Term

Implementation:

```python
sq_to_point = distances.min(dim=3).values.mean(dim=-1)
sq_to_point = (sq_to_point * exist).sum(dim=-1)
sq_to_point = sq_to_point / exist.sum(dim=-1).clamp_min(eps)
sq_to_point = sq_to_point.mean()
```

Shape path:

```text
min over input points N: [B, P, S_sq]
mean over S_sq:          [B, P]
multiply by exist:       [B, P]
sum over P:              [B]
divide by sum exist:     [B]
mean over B:             scalar
```

Formula:

```text
L_sq_to_point_b =
  sum_j exist_{b,j} * ((1/S_sq) * sum_s min_i D_{b,j,s,i})
  / max(sum_j exist_{b,j}, eps)

L_sq_to_point = mean_b L_sq_to_point_b
```

### Total SQ Regularizer

```text
L_sq = L_point_to_sq + L_sq_to_point
```

If `return_components=True`, it returns:

```text
(L_sq, {"point_to_sq": L_point_to_sq, "sq_to_point": L_sq_to_point})
```

Otherwise it returns only `L_sq`.

## Parsimony Loss

Function:

```python
assignment_parsimony_loss(assign_matrix, stabilizer=0.01)
```

Input:

```text
assign_matrix [B, N, P]
```

Assignment mass:

```text
mbar_{b,j} = (1/N) * sum_i M_{b,i,j}
```

Implementation:

```python
mass = assign_matrix.mean(dim=1)
loss = (mass + stabilizer).sqrt().mean(dim=1).pow(2).mean()
```

Formula:

```text
L_par_b = ((1/P) * sum_j sqrt(mbar_{b,j} + 0.01))^2
L_par   = mean_b L_par_b
```

The stabilizer differs slightly from the clean mathematical expression in
`PROJECT.md`. The implemented code follows the SuperDec-style stabilizer to
avoid zero-mass singularities.

## Existence Loss

Function:

```python
existence_loss(assign_matrix, exist=None, exist_logit=None, point_threshold=24.0)
```

Input:

```text
assign_matrix [B, N, P]
exist         [B, P, 1], optional
exist_logit   [B, P, 1], optional
```

Targets are derived from assignment counts:

```text
target_{b,j} = 1 if sum_i M_{b,i,j} > point_threshold else 0
```

Default:

```text
point_threshold = 24.0
```

If `exist_logit` is provided, the loss is:

```text
BCEWithLogits(exist_logit.squeeze(-1), target)
```

If only `exist` is provided, the loss is:

```text
BCE(clamp(exist.squeeze(-1), 1e-6, 1 - 1e-6), target)
```

AutoDec prefers logits when available.

## AutoDec Loss Wrapper: `autodec/losses/autodec_loss.py`

### Purpose

`AutoDecLoss` composes the reconstruction and regularization losses according
to the training phase.

Constructor:

```python
AutoDecLoss(
    phase=1,
    lambda_sq=1.0,
    lambda_par=0.06,
    lambda_exist=0.01,
    lambda_cons=0.0,
    n_sq_samples=256,
    sq_tau=1.0,
    angle_sampler=None,
    sq_regularizer=None,
    exist_point_threshold=24.0,
    active_exist_threshold=0.5,
    chamfer_eps=1e-6,
    min_backward_weight=1e-3,
)
```

`phase` may be an integer or a string such as `"phase1"`, `"phase_2"`, or
`"phase-2"`. The helper normalizes it to an integer.

### Inputs

```python
loss, metrics = loss_fn(batch, outdict)
```

Required batch key:

```text
batch["points"] [B, N, 3]
```

Required decoder output keys for reconstruction:

```text
decoded_points  [B, M, 3]
decoded_weights [B, M]
```

Additional keys are needed for phase 2:

```text
assign_matrix [B, N, P]
scale         [B, P, 3]
shape         [B, P, 2]
rotate        [B, P, 3, 3]
trans         [B, P, 3]
exist or exist_logit
```

Optional keys used for metrics and consistency:

```text
surface_points  [B, M, 3]
decoded_offsets [B, M, 3]
```

### Phase 1

For `phase < 2`, the trainable objective is only:

```text
L = L_recon
```

where:

```text
L_recon = weighted_chamfer_l2(decoded_points, batch["points"], decoded_weights)
```

The wrapper may still log metrics such as scaffold Chamfer, primitive entropy,
and active primitive count if the required keys are present, but these are not
added to the phase-1 objective.

### Phase 2

For `phase >= 2`, the objective becomes:

```text
L =
  L_recon
  + lambda_sq    * L_sq
  + lambda_par   * L_par
  + lambda_exist * L_exist
```

Each regularizer is added only if its lambda is greater than zero.

Exact implementation:

```python
loss = recon

if phase >= 2:
    if lambda_sq > 0:
        loss += lambda_sq * sq_loss
    if lambda_par > 0:
        loss += lambda_par * par_loss
    if lambda_exist > 0:
        loss += lambda_exist * exist_loss
```

### Optional Consistency Term

If `lambda_cons > 0`, training/evaluation requests `return_consistency=True`
from the model. The decoder then runs a second pass over the same sampled SQ
surface with `Z=0`:

```text
consistency_decoded_points = decoder(E_dec, Z=0)
L_cons = weighted_chamfer_l2(consistency_decoded_points, target, decoded_weights)
```

and:

```text
L += lambda_cons * L_cons
```

If `lambda_cons > 0` and `consistency_decoded_points` is absent, the loss raises
a `ValueError`. `scaffold_chamfer` remains a no-grad diagnostic metric computed
from raw `surface_points`.

### Metrics

`AutoDecLoss.forward` returns a Python metrics dictionary with detached scalar
floats.

Always logged when reconstruction keys exist:

```text
recon             total weighted Chamfer reconstruction
recon_forward     forward component from weighted_chamfer_l2
recon_backward    backward component from weighted_chamfer_l2
active_weight_sum mean_b sum_m decoded_weights_{b,m}
offset_ratio      mean ||decoded_offsets|| / mean ||surface_points||
gated_offset_ratio mean ||decoded_weights * decoded_offsets|| / mean ||surface_points||
all               final weighted objective value
```

`offset_ratio` is:

```text
mean_{b,m} ||Delta_{b,m}||_2
/
max(mean_{b,m} ||p_sq_{b,m}||_2, eps)
```

It uses `decoded_offsets` after any configured `offset_cap`/`offset_scale`, but
before multiplying by the existence gate.

`gated_offset_ratio` applies the soft existence weights first, so it measures
the displacement actually added in:

```text
decoded_points = surface_points + decoded_weights * decoded_offsets
```

When `offset_limit` exists, the loss metrics also include:

```text
offset_cap_saturation          mean abs(decoded_offsets) / offset_limit
offset_cap_saturated_fraction  fraction of offset components >= 0.95 of offset_limit
```

If `surface_points` and `decoded_weights` exist:

```text
scaffold_chamfer = weighted_chamfer_l2(surface_points, target, decoded_weights)
```

This is computed under `torch.no_grad()` in `_scaffold_chamfer`, so it is a
monitoring metric unless used separately through `lambda_cons`.

If `assign_matrix` exists:

```text
primitive_mass_entropy
```

where:

```text
mass_{b,j} = mean_i M_{b,i,j}
prob_{b,j} = mass_{b,j} / max(sum_k mass_{b,k}, eps)
entropy_b  = -sum_j prob_{b,j} * log(max(prob_{b,j}, eps))
metric     = mean_b entropy_b
```

If `exist` or `exist_logit` exists:

```text
active_primitive_count
```

where:

```text
exist = outdict["exist"] if present else sigmoid(outdict["exist_logit"])
active_primitive_count = mean_b sum_j 1[exist_{b,j} > active_exist_threshold]
```

Default:

```text
active_exist_threshold = 0.5
```

Phase-2 metrics also include:

```text
sq_loss
sq_point_to_prim
sq_prim_to_point
parsimony_loss
exist_loss
```

when their corresponding lambdas are active.

If `lambda_cons > 0` and `surface_points` is available:

```text
consistency_loss
```

is also logged.

## Comparison With Original SuperDec

Original SuperDec lives in `superdec/superdec.py`. Its forward flow is:

```text
points
  -> StackedPVConv
  -> TransformerDecoder
  -> SuperDecHead
  -> assignment matrix
  -> primitive outdict
```

It returns only the final primitive dictionary:

```text
scale
shape
rotate
trans
exist
assign_matrix
possibly extended fields
```

AutoDec changes this in several concrete ways:

1. It uses `autodec.models.heads.SuperDecHead` for `head_type="heads"` so raw
   `exist_logit` is available.
2. It keeps `point_features` and `sq_features` when `return_features=True`.
3. It adds `PartResidualProjector` to compute `residual [B, P, D]`.
4. It adds `AutoDecDecoder` to reconstruct dense point coordinates.
5. It adds AutoDec-specific losses under `autodec/losses/`.

AutoDec does not modify the `superdec/` package.

## Checkpoint Compatibility

`AutoDecEncoder.load_state_dict` is intentionally tolerant of new AutoDec-only
parameters while keeping SuperDec checkpoint names stable.

Allowed missing prefixes include:

```text
heads.tapering_head
heads.bending_k_head
heads.bending_a_head
residual_projector
```

If `clear_orientation_heads=True`, the loader removes:

```text
heads.scale_head.weight
heads.scale_head.bias
heads.shape_head.weight
heads.shape_head.bias
heads.rot_head.weight
heads.rot_head.bias
```

and allows those prefixes to be missing.

If the AutoDec head uses 6D rotation but the checkpoint has a 4D rotation head,
the loader removes:

```text
heads.rot_head.weight
heads.rot_head.bias
```

and allows the rotation head to be missing.

With `strict=True`, missing keys are accepted only if they match the allowed
prefixes. Unexpected keys still raise.

## Implemented Data Contracts

The current code expects these contracts to hold.

### Encoder Contract

Input:

```text
x [B, N, 3]
```

Output required by decoder:

```text
scale        [B, P, 3]
shape        [B, P, 2]
rotate       [B, P, 3, 3]
trans        [B, P, 3]
exist_logit  [B, P, 1]
residual     [B, P, D]
```

Output required by phase-2 losses:

```text
assign_matrix [B, N, P]
exist or exist_logit
```

### Decoder Contract

Input:

```text
encoder outdict with keys above
```

Output required by reconstruction loss:

```text
decoded_points  [B, P*S, 3]
decoded_weights [B, P*S]
```

Additional outputs useful for losses and diagnostics:

```text
surface_points   [B, P*S, 3]
decoded_offsets  [B, P*S, 3]
part_ids         [P*S]
E_dec            [B, P, 18]
```

### Loss Contract

Input:

```text
batch["points"] [B, N, 3]
outdict["decoded_points"] [B, M, 3]
outdict["decoded_weights"] [B, M]
```

Output:

```text
loss    scalar tensor
metrics dict[str, float]
```

## Shape Example With Default Project Values

Assume:

```text
B = 8
N = 4096
P = 16
H = 128
D = 64
S = 256
M = P*S = 4096
```

Encoder:

```text
points                  [8, 4096, 3]
point_features           [8, 4096, 128]
refined queries/layer    [8, 17, 128]
sq_features              [8, 16, 128]
assign_matrix            [8, 4096, 16]
scale                    [8, 16, 3]
shape                    [8, 16, 2]
rotate                   [8, 16, 3, 3]
trans                    [8, 16, 3]
exist_logit              [8, 16, 1]
exist                    [8, 16, 1]
pooled_features          [8, 16, 128]
residual                 [8, 16, 64]
```

Decoder:

```text
canonical_surface_points [8, 16, 256, 3]
surface_points_by_part   [8, 16, 256, 3]
surface_points           [8, 4096, 3]
decoded_weights          [8, 4096]
E_dec                    [8, 16, 18]
surface_position_features [8, 4096, 39]
decoder_features         [8, 4096, 128]
primitive_tokens         [8, 16, 64]
decoded_offsets          [8, 4096, 3]
decoded_points           [8, 4096, 3]
decoder_attention        [8, 4096, 16] if requested
consistency_decoded_points [8, 4096, 3] if requested
```

Loss:

```text
target points            [8, 4096, 3]
Chamfer distances        [8, 4096, 4096]
reconstruction loss      scalar
phase-2 SQ distances     [8, 16, S_sq, 4096]
phase-2 total loss       scalar
```

## What Is Not Implemented Yet

The current `autodec/` code already includes the high-level model wrapper,
training builders/trainer, evaluation entrypoint, visualizations, metrics
helpers, and checkpoint helpers. The remaining model-level gaps are:

```text
autodec/data/* re-exports
normal-alignment SQ loss
training-time Z-dropout or residual dropout
adaptive sampling per primitive
training-time paper metrics and pruned training visualizations
editing/correspondence utilities for part removal, swap, and interpolation
```

`lambda_cons` now uses the true second decoder pass with `residual = 0` when it
is enabled. It remains disabled by default in the YAML configs.

## Correctness Checklist

Use this checklist when reviewing the implementation against the intended
model.

1. Encoder input is `[B, N, 3]`.
2. `StackedPVConv` returns `[B, N, H]`.
3. Transformer query output is `[B, P + 1, H]`.
4. Primitive features drop the final query token and become `[B, P, H]`.
5. Assignment logits drop the final query token and become `[B, N, P]`.
6. `assign_matrix = softmax(assign_logits, dim=2)`.
7. AutoDec head returns `exist_logit` and `exist`.
8. Fallback heads reconstruct `exist_logit = logit(clamp(exist))`.
9. Residual pooling returns mean diagnostics with `einsum("bnp,bnh->bph")` and
   feeds mean, masked max, and variance statistics to the residual MLP.
10. Residual latents have shape `[B, P, D]`.
11. Decoder primitive code `E_dec` has 18 values, not 12.
12. Surface sampler detaches only the angle-selection inputs, not the surface
    coordinate computation.
13. Signed powers preserve exact zero coordinates because `sign(0) = 0`.
14. Surface points are transformed by `R * canonical + t`.
15. Existence weights are `sigmoid(exist_logit / tau)`.
16. Existence weights are not multiplied into surface coordinates.
17. Point decoder features use `[projected Fourier surface position, projected E_dec, projected residual, projected gate]` by default.
18. Primitive attention tokens are `[projected E_dec, projected residual]`.
19. Cross-attention queries are sampled points; keys and values are primitives.
20. Offsets are gated in `decoded_points = surface_points + w * offsets`.
21. Reconstruction forward Chamfer is weighted by decoded weights.
22. Reconstruction backward Chamfer divides distances by decoded weights before
    nearest-neighbor selection.
23. SQ regularizer uses sampled-surface Chamfer-L2 and omits normal alignment.
24. Parsimony uses `((mean_j sqrt(mbar_j + 0.01))^2)` averaged across batch.
25. Existence targets are derived from assignment count threshold `24.0` by
    default.
26. Phase 1 objective is reconstruction only unless `lambda_cons > 0`.
27. Phase 2 objective adds SQ, parsimony, and existence terms according to
    their lambdas.
28. Consistency loss uses `consistency_decoded_points` from a second decoder
    pass with `Z=0`; scaffold Chamfer is only a diagnostic metric.
