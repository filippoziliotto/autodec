# losses

This folder contains all implemented AutoDec losses. It is independent from
`superdec.loss.loss.SuperDecLoss` because the SuperDec loss currently hardcodes
CUDA movement in places and expects normals for some paths.

Files:

```text
__init__.py
chamfer.py
sq_regularizer.py
autodec_loss.py
```

## `__init__.py`

Exports:

```text
AutoDecLoss
SQRegularizer
assignment_parsimony_loss
existence_loss
weighted_chamfer_l2
```

This makes the common imports available from:

```python
from autodec.losses import AutoDecLoss, weighted_chamfer_l2
```

## `chamfer.py`

Defines:

```text
weighted_chamfer_l2
```

Private helper:

```text
_check_chamfer_inputs
```

### `_check_chamfer_inputs`

Purpose:

Validate and normalize Chamfer input shapes.

Expected input:

```text
pred    [B, M, 3]
target  [B, N, 3]
weights [B, M] or [B, M, 1]
```

Behavior:

- Raises if `pred` is not rank 3 with final dimension 3.
- Raises if `target` is not rank 3 with final dimension 3.
- Squeezes `weights [B, M, 1]` to `[B, M]`.
- Raises if weights are not `[B, M]`.
- Raises if batch sizes do not match.
- Raises if `pred` and `weights` do not match on `[B, M]`.

### `weighted_chamfer_l2`

Purpose:

Compare fixed-size decoded points to target points while respecting soft
existence weights.

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
pred    [B, M, 3]
target  [B, N, 3]
weights [B, M] or [B, M, 1]
```

Distance tensor:

```text
D[b, m, n] = ||pred[b, m] - target[b, n]||_2^2
D shape = [B, M, N]
```

Forward term:

```text
L_fwd_b =
  sum_m max(w[b,m], eps) * min_n D[b,m,n]
  / sum_m max(w[b,m], eps)
```

Backward term:

```text
D_weighted[b,m,n] = D[b,m,n] / max(w[b,m], min_backward_weight)
L_bwd_b = mean_n min_m D_weighted[b,m,n]
```

Total:

```text
L = mean_b L_fwd_b + mean_b L_bwd_b
```

Why the backward division exists:

Without it, inactive scaffold points could satisfy target coverage during
training. Dividing by the weight before nearest-neighbor selection makes
low-weight predictions unattractive nearest neighbors.

Return behavior:

```text
return_components=False -> loss
return_components=True  -> (loss, {"forward": ..., "backward": ...})
```

## `sq_regularizer.py`

Defines:

```text
SQRegularizer
assignment_parsimony_loss
existence_loss
```

Private helpers:

```text
_batch_points
_exist_logit
```

### `_batch_points`

Moves `batch["points"]` to the same device and dtype as a reference tensor.

Input:

```text
batch["points"] [B, N, 3]
reference       tensor with desired device and dtype
```

Output:

```text
points [B, N, 3]
```

### `_exist_logit`

Returns `outdict["exist_logit"]` if available. Otherwise reconstructs logits
from `outdict["exist"]`:

```text
logit(clamp(exist, 1e-6, 1 - 1e-6))
```

### `assignment_parsimony_loss`

Purpose:

SuperDec-style sparsity surrogate that encourages assignments to use fewer
primitive slots.

Input:

```text
assign_matrix [B, N, P]
```

Assignment mass:

```text
mbar[b, j] = mean_i assign_matrix[b, i, j]
```

Loss:

```text
L_par = mean_b ((1/P) * sum_j sqrt(mbar[b,j] + 0.01))^2
```

The `0.01` stabilizer matches the current SuperDec implementation style and
avoids unstable behavior at zero primitive mass.

### `existence_loss`

Purpose:

Train primitive existence predictions from assignment mass.

Signature:

```python
existence_loss(assign_matrix, exist=None, exist_logit=None, point_threshold=24.0)
```

Target:

```text
target[b, j] = 1 if sum_i assign_matrix[b, i, j] > point_threshold else 0
```

Default:

```text
point_threshold = 24.0
```

If `exist_logit` is provided:

```text
BCEWithLogits(exist_logit.squeeze(-1), target)
```

If only `exist` is provided:

```text
BCE(clamp(exist.squeeze(-1), 1e-6, 1 - 1e-6), target)
```

If neither is provided, the function raises `ValueError`.

### `SQRegularizer`

Purpose:

Sample predicted superquadric surfaces and fit them to the input point cloud
with a bidirectional Chamfer-L2 term.

Constructor:

```python
SQRegularizer(n_samples=256, tau=1.0, angle_sampler=None, eps=1e-6)
```

Submodule:

```text
surface_sampler = SQSurfaceSampler(n_samples=n_samples, tau=tau, angle_sampler=angle_sampler)
```

Required batch key:

```text
batch["points"] [B, N, 3]
```

Required outdict keys:

```text
scale         [B, P, 3]
shape         [B, P, 2]
rotate        [B, P, 3, 3]
trans         [B, P, 3]
assign_matrix [B, N, P]
exist or exist_logit
```

Surface samples:

```text
surface_points [B, P, S_sq, 3]
```

Distance tensor:

```text
D[b, j, s, i] = ||surface_points[b,j,s] - points[b,i]||_2^2
D shape = [B, P, S_sq, N]
```

Point-to-SQ term:

```text
L_point_to_sq =
  mean_b mean_i sum_j assign_matrix[b,i,j] * min_s D[b,j,s,i]
```

SQ-to-point term:

```text
L_sq_to_point_b =
  sum_j exist[b,j] * mean_s min_i D[b,j,s,i]
  / max(sum_j exist[b,j], eps)
```

Total:

```text
L_sq = L_point_to_sq + mean_b L_sq_to_point_b
```

Return behavior:

```text
return_components=False -> L_sq
return_components=True  -> (L_sq, {"point_to_sq": ..., "sq_to_point": ...})
```

This regularizer intentionally omits normal alignment for v1.

## `autodec_loss.py`

Defines:

```text
AutoDecLoss
```

Private helpers:

```text
_metric_value
_phase_number
_target_points
_primitive_mass_entropy
_active_primitive_count
_offset_ratio
```

### `_metric_value`

Converts a scalar tensor or number into a detached Python `float`.

Used so the returned `metrics` dictionary is log-friendly and does not keep
autograd graph references.

### `_phase_number`

Normalizes `phase` input.

Accepted examples:

```text
1
2
"phase1"
"phase_2"
"phase-2"
```

Returns:

```text
int phase
```

### `_target_points`

Returns `batch["points"]` on the same device and dtype as a reference tensor.

### `_primitive_mass_entropy`

Computes entropy over average primitive assignment mass.

Mass:

```text
mass[b,j] = mean_i assign_matrix[b,i,j]
prob[b,j] = mass[b,j] / sum_k mass[b,k]
```

Entropy:

```text
entropy = -sum_j prob[b,j] * log(prob[b,j])
```

Returns mean entropy across batch.

### `_active_primitive_count`

Uses `outdict["exist"]` if present, otherwise `sigmoid(outdict["exist_logit"])`.

Metric:

```text
mean_b sum_j 1[exist[b,j] > active_exist_threshold]
```

Default threshold:

```text
0.5
```

### `_offset_ratio`

Diagnostic for residual/offset dominance:

```text
mean ||decoded_offsets||_2 / mean ||surface_points||_2
```

If either key is missing, returns a zero scalar on the decoded point device.

### `AutoDecLoss`

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

Required inputs to `forward(batch, outdict)`:

```text
batch["points"]             [B, N, 3]
outdict["decoded_points"]   [B, M, 3]
outdict["decoded_weights"]  [B, M]
```

When `lambda_cons > 0`, also requires:

```text
outdict["consistency_decoded_points"] [B, M, 3]
```

Phase-2 also requires:

```text
assign_matrix [B, N, P]
scale         [B, P, 3]
shape         [B, P, 2]
rotate        [B, P, 3, 3]
trans         [B, P, 3]
exist or exist_logit
```

#### Reconstruction loss

Always computed:

```text
L_recon = weighted_chamfer_l2(decoded_points, target_points, decoded_weights)
```

Metrics:

```text
recon
recon_forward
recon_backward
```

#### Phase 1 objective

For `phase < 2`:

```text
L = L_recon
```

SQ/parsimony/existence terms are not added, even if their lambdas are nonzero.

#### Phase 2 objective

For `phase >= 2`:

```text
L = L_recon
```

Then conditionally:

```text
if lambda_sq > 0:
    L += lambda_sq * L_sq

if lambda_par > 0:
    L += lambda_par * L_par

if lambda_exist > 0:
    L += lambda_exist * L_exist
```

#### Consistency term

If `lambda_cons > 0`, the trainer/evaluator request a second decoder pass with
the residual latent zeroed:

```text
Z_cons = zeros_like(Z)
consistency_decoded_points = decoder(E_dec, Z_cons)
L_cons = weighted_chamfer_l2(
    consistency_decoded_points,
    target_points,
    decoded_weights,
)
L += lambda_cons * L_cons
```

This is the intended no-residual consistency loss. It still uses the learned
offset decoder; it just removes residual information from the decoder input.

If `lambda_cons > 0` and `consistency_decoded_points` is missing from the
outdict, `AutoDecLoss` raises a `ValueError` instead of silently falling back to
scaffold Chamfer.

`scaffold_chamfer` remains a diagnostic metric:

```text
weighted_chamfer_l2(surface_points, target_points, decoded_weights)
```

It is computed under `torch.no_grad()` and is not added to the objective.

#### Metrics

Always possible when relevant keys exist:

```text
recon
recon_forward
recon_backward
active_weight_sum
offset_ratio
scaffold_chamfer
primitive_mass_entropy
active_primitive_count
all
```

Phase-2 regularizer metrics:

```text
sq_loss
sq_point_to_prim
sq_prim_to_point
parsimony_loss
exist_loss
```

Optional consistency metric:

```text
consistency_loss
```
