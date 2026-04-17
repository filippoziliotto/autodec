# models

This folder tests `autodec/models/`.

Files:

```text
test_heads.py
test_residual.py
test_offset_decoder.py
```

## `test_heads.py`

Tests:

```text
autodec.models.heads.SuperDecHead
```

### `test_head_returns_exist_logit_and_probability`

Creates:

```python
ctx = SimpleNamespace(rotation6d=False, extended=False)
head = SuperDecHead(emb_dims=8, ctx=ctx)
```

Input:

```text
torch.randn(2, 3, 8)
```

Meaning:

```text
B = 2
P = 3
H = 8
```

Checks:

```text
exist_logit shape [2, 3, 1]
exist shape [2, 3, 1]
exist == sigmoid(exist_logit)
rotation_quat shape [2, 3, 4]
rotate shape [2, 3, 3, 3]
```

Purpose:

Ensure the AutoDec-owned head preserves SuperDec primitive outputs and adds the
raw existence logit required by decoder gates and BCE-with-logits losses.

## `test_residual.py`

Tests:

```text
autodec.models.residual.PartResidualProjector
```

### `test_part_residual_projector_pools_point_features_by_assignment`

Creates:

```text
point_features [1, 3, 2]
assign_matrix  [1, 3, 2]
```

Concrete values:

```text
point 0 -> [1, 2], assigned to primitive 0
point 1 -> [3, 4], assigned to primitive 1
point 2 -> [5, 6], assigned to primitive 0
```

Expected pooled output:

```text
primitive 0 = ([1,2] + [5,6]) / 2 = [3,4]
primitive 1 = [3,4] / 1 = [3,4]
```

Check:

```text
pooled == [[[3,4], [3,4]]]
```

Purpose:

Verify the exact weighted-pooling formula:

```text
pooled = einsum("bnp,bnh->bph", M, F_PC) / sum_i M
```

### `test_part_residual_projector_returns_residual_and_pooled_features`

Creates:

```text
projector = PartResidualProjector(feature_dim=4, residual_dim=5, hidden_dim=8)
sq_features    [2, 3, 4]
point_features [2, 7, 4]
assign_matrix  [2, 7, 3]
```

Calls with:

```python
return_pooled=True
```

Checks:

```text
residual shape [2, 3, 5]
pooled shape [2, 3, 4]
```

Purpose:

Confirm the module returns both the residual latent and pooled local feature
when requested by `AutoDecEncoder`.

## `test_offset_decoder.py`

Tests:

```text
autodec.models.offset_decoder.CrossAttentionOffsetDecoder
autodec.models.offset_decoder.build_offset_decoder
```

### `test_cross_attention_offset_decoder_returns_offsets_and_attention_weights`

Creates:

```text
CrossAttentionOffsetDecoder(
    point_in_dim=10,
    primitive_in_dim=8,
    hidden_dim=12,
    n_heads=3,
)
```

Inputs:

```text
point_features   [2, 5, 10]
primitive_tokens [2, 3, 8]
```

Meaning:

```text
B = 2
M = 5 sampled points
P = 3 primitive tokens
```

Calls:

```python
return_attention=True
```

Checks:

```text
offsets shape [2, 5, 3]
attention shape [2, 5, 3]
attention sums to 1 over primitive dimension
```

Purpose:

Verify point-to-primitive cross-attention has the expected external contract:
one 3D offset per sampled point and one averaged attention distribution over
primitive tokens per point.

### `test_build_offset_decoder_builds_cross_attention_decoder`

Calls:

```python
build_offset_decoder(
    decoder_type="cross_attention",
    point_in_dim=10,
    primitive_in_dim=8,
    hidden_dim=12,
    n_heads=3,
)
```

Checks:

```text
isinstance(decoder, CrossAttentionOffsetDecoder)
```

Purpose:

Protect the factory API used by `AutoDecDecoder`.

