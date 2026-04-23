# losses

## Purpose

`gendec/losses/` contains the token-space flow path construction and the training objectives used by Phase 1 and Phase 2.

## Maintenance Contract

If the interpolation path, auxiliary objectives, or exported loss API change, this file must be updated in the same change.

## Files

### `__init__.py`

- Re-export surface for the loss layer.
- Exposes `FlowMatchingLoss`, `JointFlowMatchingLoss`, `build_flow_batch`, `build_flow_path`, and `reconstruct_clean_tokens`.

### `flow_matching.py`

- High-level loss modules used by training and evaluation.
- `FlowMatchingLoss`:
  - `__init__(lambda_flow=1.0, lambda_exist=0.05, exist_channel=-1)`: stores the Phase 1 flow-loss weight, existence-loss weight, and existence-logit channel index.
  - `_per_sample_metrics(batch, v_hat)`: computes per-sample flow loss and optional per-sample existence BCE, then combines them into total per-sample loss using the configured weights.
  - `forward(batch, v_hat, return_per_sample=False)`: returns the scalar Phase 1 training loss plus aggregated metric scalars, and optionally the unreduced per-sample metric tensors.
- `JointFlowMatchingLoss`:
  - `__init__(explicit_dim=TOKEN_DIM, lambda_e=1.0, lambda_z=1.0, lambda_exist=0.05, exist_channel=-1)`: stores the split explicit/residual weights and the existence-logit channel index.
  - `_split_velocity(v)`: splits a concatenated joint velocity into explicit and residual slices.
  - `_per_sample_metrics(batch, v_hat_e, v_hat_z, v_hat)`: computes per-sample explicit flow loss, residual flow loss, optional existence BCE, and the weighted total loss.
  - `forward(batch, v_hat_e, v_hat_z, v_hat=None, return_per_sample=False)`: returns the scalar Phase 2 training loss plus aggregated metric scalars, and optionally the unreduced per-sample metric tensors.

### `objectives.py`

- Lower-level objective helpers used by the loss modules.
- `per_sample_flow_mse(v_hat, velocity_target)`: computes one MSE scalar per sample over all primitive-token channels.
- `reconstruct_clean_tokens(batch, v_hat)`: reconstructs `E0_hat = Et - t * v_hat`.
- `unnormalize_exist_logits(tokens, mean, std, exist_channel)`: converts normalized existence logits back into raw logit scale.
- `per_sample_exist_bce(batch, v_hat, exist_channel)`: computes one existence BCE scalar per sample from reconstructed clean tokens.

### `path.py`

- Analytic straight-line interpolation path for flow matching.
- `build_flow_batch(e0, e1=None, t=None)`: samples or accepts `E1` and `t`, then builds `E0`, `E1`, `Et`, `velocity_target`, and `t`.
