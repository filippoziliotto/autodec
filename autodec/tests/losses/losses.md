# losses

This folder tests `autodec/losses/`.

Files:

```text
test_chamfer.py
test_sq_regularizer.py
test_autodec_loss.py
```

## `test_chamfer.py`

Tests:

```text
autodec.losses.chamfer.weighted_chamfer_l2
```

### `test_weighted_chamfer_forward_does_not_let_inactive_outlier_dominate`

Inputs:

```text
pred    [[[0,0,0], [10,0,0]]]
target  [[[0,0,0]]]
weights [[1, 0]]
```

Expected behavior:

The inactive outlier at `[10,0,0]` should not dominate the forward term.

Checks:

```text
loss < 1e-3
forward < 1e-3
backward == 0
```

Purpose:

Verify low-weight predictions are effectively ignored in the prediction-to-
target direction.

### `test_weighted_chamfer_backward_discourages_low_weight_matches`

Inputs:

```text
pred    [[[0.1,0,0], [0.5,0,0]]]
target  [[[0,0,0]]]
weights [[0.001, 1.0]]
min_backward_weight = 0.001
```

Raw squared distances:

```text
0.1^2 = 0.01
0.5^2 = 0.25
```

Backward-weighted distances:

```text
0.01 / 0.001 = 10
0.25 / 1.0   = 0.25
```

Expected nearest weighted distance:

```text
0.25
```

Check:

```text
components["backward"] == 0.25
```

Purpose:

Verify the backward Chamfer term avoids explaining target points with
low-weight inactive predictions.

## `test_sq_regularizer.py`

Tests:

```text
SQRegularizer
assignment_parsimony_loss
existence_loss
```

### Local `FixedAngleSampler`

Returns one deterministic sample:

```text
eta = 0
omega = 0
```

for every batch item and primitive.

### `_sq_outdict`

Builds one simple primitive:

```text
scale         ones [1, 1, 3]
shape         ones [1, 1, 2]
rotate        identity [1, 1, 3, 3]
trans         zeros [1, 1, 3]
exist_logit   20 [1, 1, 1]
exist         1 [1, 1, 1]
assign_matrix ones [1, 2, 1]
```

With `eta=0`, `omega=0`, the sampled SQ point is:

```text
[1, 0, 0]
```

### `test_sq_regularizer_uses_sampled_surface_chamfer_l2`

Batch points:

```text
[[[1,0,0], [2,0,0]]]
```

Expected point-to-SQ:

```text
point [1,0,0] -> distance 0
point [2,0,0] -> squared distance 1
mean = 0.5
```

Expected SQ-to-point:

```text
sample [1,0,0] -> nearest point [1,0,0] -> distance 0
```

Checks:

```text
point_to_sq == 0.5
sq_to_point == 0.0
loss == 0.5
```

Purpose:

Verify the sampled-surface Chamfer-L2 formulas.

### `test_assignment_parsimony_prefers_concentrated_mass`

Compares:

```text
concentrated = all assignment mass on primitive 0
balanced     = mass split evenly across two primitives
```

Check:

```text
assignment_parsimony_loss(concentrated)
<
assignment_parsimony_loss(balanced)
```

Purpose:

Verify the sparsity surrogate prefers fewer active primitive slots.

### `test_existence_loss_uses_assignment_mass_targets`

Assignment matrix has four points and two primitives:

```text
primitive 0 receives mass 3
primitive 1 receives mass 1
```

With `point_threshold=2.0`, target is:

```text
[1, 0]
```

Good logits:

```text
[20, -20]
```

Bad logits:

```text
[-20, 20]
```

Check:

```text
good loss < bad loss
```

Purpose:

Verify existence supervision is derived from assignment mass and uses logits
correctly.

## `test_autodec_loss.py`

Tests:

```text
autodec.losses.autodec_loss.AutoDecLoss
```

### Local `FixedAngleSampler`

Returns one deterministic SQ sample:

```text
eta = 0
omega = 0
```

### `_batch`

Returns:

```text
batch["points"] = [[[1,0,0], [2,0,0]]]
```

### `_outdict`

Builds a minimal combined decoder/encoder output:

```text
decoded_points  [[[1,0,0]]]
decoded_weights [[1]]
surface_points  [[[1,0,0]]]
decoded_offsets zeros [1, 1, 3]
scale           ones [1, 1, 3]
shape           ones [1, 1, 2]
rotate          identity [1, 1, 3, 3]
trans           zeros [1, 1, 3]
exist_logit     20 [1, 1, 1]
exist           1 [1, 1, 1]
assign_matrix   ones [1, 2, 1]
```

### `test_autodec_loss_phase1_uses_reconstruction_only`

Constructs `AutoDecLoss` with:

```text
phase = 1
lambda_sq = 10
lambda_par = 10
lambda_exist = 10
```

Checks:

```text
loss.item() == metrics["recon"]
"sq_loss" not in metrics
metrics["all"] == metrics["recon"]
```

Purpose:

Verify phase 1 ignores regularizer lambdas and uses reconstruction only.

### `test_autodec_loss_is_exported_from_package_root`

Imports:

```python
from autodec import AutoDecLoss
```

Checks:

```text
AutoDecLoss.__name__ == "AutoDecLoss"
```

Purpose:

Protect the root package export.

### `test_autodec_loss_phase2_composes_regularizers_and_metrics`

Constructs:

```text
phase = 2
lambda_sq = 2
lambda_par = 3
lambda_exist = 5
exist_point_threshold = 1
```

Expected objective:

```text
recon
+ 2 * sq_loss
+ 3 * parsimony_loss
+ 5 * exist_loss
```

Checks:

```text
abs(loss.item() - expected) < 1e-6
offset_ratio == 0
active_primitive_count == 1
primitive_mass_entropy exists
scaffold_chamfer exists
```

Purpose:

Verify phase 2 composes weighted loss terms and logs diagnostic metrics.

