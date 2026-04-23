# models

## Purpose

`gendec/models/` contains the Phase 1 and Phase 2 neural networks plus the shared geometry helpers: token projection, attention blocks, time embeddings, rotation conversions, and the top-level Set Transformer flow models.

## Maintenance Contract

If the model architecture, tensor contract, or helper geometry conversions change, this file must be updated in the same change.

## Files

### `__init__.py`

- Re-export surface for the model layer.
- Exposes `ClassConditioning`, `GlobalToken`, `SetTransformerBlock`, `SetTransformerFlowModel`, `JointSetTransformerFlowModel`, `TokenProjection`, `VelocityHead`, `matrix_to_rot6d`, and `rot6d_to_matrix`.

### `components.py`

- Reusable model building blocks.
- `TokenProjection`: token projection MLP from raw token space into hidden space.
- `GlobalToken`: learned global token expanded to `[B,1,H]`.
- `VelocityHead`: token-wise velocity prediction head.
- `ClassConditioning`: learned class embedding plus optional projection into the transformer hidden width.
- `SetTransformerBlock`: one self-attention + feed-forward transformer block.

### `rotation.py`

- Rotation-representation conversion helpers.
- `matrix_to_rot6d(matrix)`: flattens the first two rotation columns into the 6D representation.
- `rot6d_to_matrix(rot6d)`: converts 6D rotation vectors back into valid rotation matrices using Gram-Schmidt orthogonalization.

### `set_transformer_flow.py`

- Top-level flow networks.
- `SetTransformerFlowModel`:
  - `__init__(token_dim=15, hidden_dim=256, n_blocks=6, n_heads=8, dropout=0.0, conditioning_enabled=False, num_classes=1, class_embed_dim=None)`: wires token projection, time embedding, optional class conditioning, global token, attention blocks, and the single velocity head for Phase 1.
  - `conditioning_active`: becomes `True` only when conditioning is enabled and `num_classes > 1`.
  - `forward(et, t, category_index=None)`: predicts token-space velocities from interpolated scaffold tokens, scalar times, and optional class ids.
- `JointSetTransformerFlowModel`:
  - `__init__(explicit_dim=15, residual_dim=64, hidden_dim=384, n_blocks=6, n_heads=8, dropout=0.0, conditioning_enabled=False, num_classes=1, class_embed_dim=None)`: wires the shared backbone, optional class conditioning, and two output heads for explicit and residual velocities.
  - `conditioning_active`: becomes `True` only when conditioning is enabled and `num_classes > 1`.
  - `forward(tt, t, category_index=None)`: predicts `v_hat_e`, `v_hat_z`, and the concatenated `v_hat` from interpolated joint tokens, scalar times, and optional class ids.

### `time_embedding.py`

- Time-conditioning module.
- `SinusoidalTimeEmbedding`:
  - `__init__(hidden_dim, embedding_dim=128)`: configures the sinusoidal basis size and projection MLP.
  - `forward(t)`: converts scalar time inputs into `hidden_dim` embeddings.
