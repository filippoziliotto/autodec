# sampling

This folder contains superquadric surface sampling utilities for AutoDec.

Files:

```text
__init__.py
sq_surface.py
```

## `__init__.py`

Exports:

```python
from autodec.sampling.sq_surface import SQSurfaceSample, SQSurfaceSampler
```

Public names:

```text
SQSurfaceSample
SQSurfaceSampler
```

## `sq_surface.py`

Defines:

```text
SQSurfaceSample
SQSurfaceSampler
```

Purpose:

Sample surface coordinates from predicted superquadric parameters and compute
soft existence weights for sampled points.

## `SQSurfaceSample`

Dataclass returned by `SQSurfaceSampler.forward`.

Fields:

```text
canonical_points [B, P, S, 3]
surface_points   [B, P, S, 3]
flat_points      [B, P*S, 3]
part_ids         [P*S]
weights          [B, P*S]
```

Meaning:

- `canonical_points`: points before rotation and translation.
- `surface_points`: world-space points grouped by primitive.
- `flat_points`: world-space points flattened in primitive-major order.
- `part_ids`: parent primitive index for every flattened point.
- `weights`: existence gate repeated for every point from a primitive.

## `SQSurfaceSampler`

Constructor:

```python
SQSurfaceSampler(n_samples=256, tau=1.0, angle_sampler=None, eps=1e-6)
```

Parameters:

```text
n_samples     S, number of samples per primitive
tau           existence-gate temperature
angle_sampler optional injected sampler for tests or custom sampling
eps           numerical floor used in signed powers
```

The sampler defensively clamps shape exponents to `[0.1, 2.0]` before calling
the angle sampler and before evaluating signed-power surface coordinates. Normal
AutoDec head outputs are already in this range, but this protects manually built
or externally loaded outdicts.

### Required input keys

`forward(outdict)` requires:

```text
scale        [B, P, 3]
shape        [B, P, 2]
rotate       [B, P, 3, 3]
trans        [B, P, 3]
exist_logit  [B, P, 1]
```

### Angle selection

If `angle_sampler` was injected, that object is used. It must expose:

```python
sample_on_batch(scale_numpy, shape_numpy)
```

and return:

```text
etas   [B, P, S]
omegas [B, P, S]
```

If no sampler is injected, AutoDec imports SuperDec's equal-distance sampler:

```python
from superdec.loss.sampler import EqualDistanceSamplerSQ
```

and creates:

```python
EqualDistanceSamplerSQ(n_samples=S, D_eta=0.05, D_omega=0.05)
```

The angle sampler receives detached NumPy arrays:

```python
scale.detach().cpu().numpy()
shape.detach().cpu().numpy()
```

Therefore angle choice is not differentiable with respect to scale or shape.
The coordinate computation after the angles are returned is differentiable.

### Signed power

The helper `_signed_power(value, exponent)` implements:

```text
signed_power(v, e) = sign(v) * max(abs(v), eps)^e
```

This is necessary for fractional superquadric exponents across all quadrants.

Important zero behavior:

```text
sign(0) = 0
```

So an exact zero coordinate remains zero even though `abs(v)` is clamped before
the power.

### Canonical surface equation

For each primitive:

```text
scale = [sx, sy, sz]
shape = [e1, e2]
eta   [B, P, S]
omega [B, P, S]
```

The code computes:

```text
x = sx * signed_power(cos(eta), e1) * signed_power(cos(omega), e2)
y = sy * signed_power(cos(eta), e1) * signed_power(sin(omega), e2)
z = sz * signed_power(sin(eta), e1)
```

Output:

```text
canonical_points [B, P, S, 3]
```

### World-space transform

The sampler applies:

```text
p_world = R * p_canonical + t
```

Code shape behavior:

```text
rotate.unsqueeze(2)      [B, P, 1, 3, 3]
canonical.unsqueeze(-1)  [B, P, S, 3, 1]
surface_points           [B, P, S, 3]
```

Then:

```python
surface = surface + trans.unsqueeze(2)
```

### Flattening and part IDs

Flattened points are primitive-major:

```text
flat index = primitive_index * S + sample_index
```

`part_ids` is:

```text
[0, ..., 0, 1, ..., 1, ..., P-1, ..., P-1]
```

where each primitive index appears `S` times.

Shape:

```text
part_ids [P*S]
```

### Existence weights

Primitive gate:

```text
w_j = sigmoid(exist_logit_j / tau)
```

The gate is repeated for each sampled point:

```text
weights[b, j*S + s] = w[b, j]
```

Shape:

```text
weights [B, P*S]
```

Important:

The sampler does not multiply points by `weights`. The coordinates stay
geometrically valid. The decoder uses the weights later to gate offsets:

```text
decoded_points = surface_points + weights * offsets
```
