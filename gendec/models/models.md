# models

## Purpose

`gendec/models/` contains the Phase 1 neural network and geometry helpers: token projection, attention blocks, time embeddings, rotation conversions, and the top-level Set Transformer flow model.

## Maintenance Contract

If the model architecture, tensor contract, or helper geometry conversions change, this file must be updated in the same change.

## Files

### `__init__.py`

- Re-export surface for the model layer.
- Exposes `GlobalToken`, `SetTransformerBlock`, `SetTransformerFlowModel`, `TokenProjection`, `VelocityHead`, `matrix_to_rot6d`, and `rot6d_to_matrix`.

### `components.py`

- Reusable model building blocks.
- `TokenProjection`:
  - `__init__(token_dim, hidden_dim)`: builds the token projection MLP.
  - `forward(tokens)`: maps `[B, P, token_dim]` tokens to `[B, P, hidden_dim]`.
- `GlobalToken`:
  - `__init__(hidden_dim)`: creates a learned global token parameter.
  - `forward(batch_size)`: expands the learned token to `[B, 1, hidden_dim]`.
- `VelocityHead`:
  - `__init__(hidden_dim, token_dim)`: builds the final token-wise prediction head.
  - `forward(hidden)`: maps hidden primitive states back to token-dimension velocities.
- `SetTransformerBlock`:
  - `__init__(hidden_dim, n_heads, dropout=0.0)`: builds one full self-attention block plus feed-forward sublayer.
  - `forward(hidden)`: applies attention, residuals, and feed-forward updates to `[B, P+1, hidden_dim]`.

### `rotation.py`

- Rotation-representation conversion helpers.
- `matrix_to_rot6d(matrix)`: flattens the first two rotation columns into the 6D representation.
- `rot6d_to_matrix(rot6d)`: converts 6D rotation vectors back into valid rotation matrices using Gram-Schmidt orthogonalization.

### `set_transformer_flow.py`

- Top-level Phase 1 network.
- `SetTransformerFlowModel`:
  - `__init__(token_dim=15, hidden_dim=256, n_blocks=6, n_heads=8, dropout=0.0)`: wires token projection, time embedding, global token, attention blocks, and velocity head.
  - `forward(et, t)`: predicts token-space velocities from interpolated tokens and scalar times.

### `time_embedding.py`

- Time-conditioning module.
- `SinusoidalTimeEmbedding`:
  - `__init__(hidden_dim, embedding_dim=128)`: configures the sinusoidal basis size and projection MLP.
  - `forward(t)`: converts scalar time inputs into `hidden_dim` embeddings.
