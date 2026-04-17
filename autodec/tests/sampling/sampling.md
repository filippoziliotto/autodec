# sampling

This folder tests `autodec/sampling/`.

Files:

```text
test_sq_surface.py
```

## `test_sq_surface.py`

Tests:

```text
autodec.sampling.sq_surface.SQSurfaceSampler
```

The tests inject a fixed angle sampler so they do not depend on SuperDec's
compiled equal-distance sampler.

## Local Helpers

### `FixedAngleSampler`

Constructor:

```python
FixedAngleSampler(etas, omegas)
```

Stores torch tensors and returns NumPy arrays from:

```python
sample_on_batch(scale, shape)
```

Returned shapes:

```text
etas   [B, P, S]
omegas [B, P, S]
```

The helper repeats the stored 1D angle tensors across the requested batch and
primitive dimensions.

### `_outdict`

Signature:

```python
_outdict(exist_logit=-4.0)
```

Returns:

```text
scale       ones [1, 2, 3], requires_grad=True
shape       ones [1, 2, 2], requires_grad=True
rotate      identity [1, 2, 3, 3]
trans       [[[1,0,0], [0,2,0]]], requires_grad=True
exist_logit constant [1, 2, 1]
```

This gives two simple primitives with known translations.

## Tests

### `test_sq_surface_sampler_returns_points_weights_and_part_ids`

Creates:

```text
SQSurfaceSampler(n_samples=2, tau=2.0, angle_sampler=fixed)
```

Angles:

```text
eta   = [0, 0]
omega = [0, pi/2]
```

Uses default `_outdict(exist_logit=-4.0)`.

Expected gate:

```text
sigmoid(exist_logit / tau) = sigmoid(-4 / 2) = sigmoid(-2)
```

Checks:

```text
flat_points shape [1, 4, 3]
surface_points shape [1, 2, 2, 3]
weights shape [1, 4]
part_ids == [0, 0, 1, 1]
weights == sigmoid(-2) repeated 4 times
```

Purpose:

Verify shape contracts, primitive-major flattening, part IDs, and temperature
gating.

### `test_sq_surface_sampler_does_not_scale_coordinates_by_existence_weight`

Creates a sampler with one sample:

```text
eta = 0
omega = 0
```

Uses:

```text
exist_logit = -20
```

For primitive 0:

```text
scale = [1,1,1]
translation = [1,0,0]
canonical point at eta=0, omega=0 is [1,0,0]
world point is [2,0,0]
```

Checks:

```text
flat_points[0,0] == [2,0,0]
weight < 1e-6
```

Purpose:

Protect the critical rule that existence weights do not multiply coordinates.

### `test_sq_surface_sampler_keeps_gradients_to_sq_parameters`

Uses one fixed sample and computes:

```text
sample.flat_points.square().mean().backward()
```

Checks gradients exist for:

```text
scale
shape
trans
```

Purpose:

Verify the sampled coordinate computation is differentiable with respect to SQ
parameters after fixed angles are chosen.

This does not test differentiability of angle selection itself. The production
sampler intentionally detaches scale and shape before calling the SuperDec
NumPy/C++ equal-distance sampler.

