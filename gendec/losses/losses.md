# losses

## Purpose

`gendec/losses/` contains the token-space flow path construction and the training objectives used by Phase 1.

## Maintenance Contract

If the interpolation path, auxiliary objectives, or exported loss API change, this file must be updated in the same change.

## Files

### `__init__.py`

- Re-export surface for the loss layer.
- Exposes `FlowMatchingLoss`, `build_flow_batch`, `build_flow_path`, and `reconstruct_clean_tokens`.

### `flow_matching.py`

- High-level loss module used by training and evaluation.
- `FlowMatchingLoss`:
  - `__init__(lambda_flow=1.0, lambda_exist=0.05, exist_channel=-1)`: stores the flow-loss weight, existence-loss weight, and the existence-logit channel index.
  - `_per_sample_metrics(batch, v_hat)`: computes per-sample flow loss and optional per-sample existence BCE, then combines them into total per-sample loss using the configured weights.
  - `forward(batch, v_hat, return_per_sample=False)`: returns the scalar training loss plus aggregated metric scalars, and optionally the unreduced per-sample metric tensors.
- `__all__`: exports `FlowMatchingLoss` and `build_flow_batch`.

### `objectives.py`

- Lower-level objective helpers used by `FlowMatchingLoss`.
- `per_sample_flow_mse(v_hat, velocity_target)`: computes one MSE scalar per sample over all primitive-token channels.
- `reconstruct_clean_tokens(batch, v_hat)`: reconstructs `E0_hat = Et - t * v_hat`.
- `unnormalize_exist_logits(tokens, mean, std, exist_channel)`: converts normalized existence logits back into raw logit scale.
- `per_sample_exist_bce(batch, v_hat, exist_channel)`: computes one existence BCE scalar per sample from reconstructed clean tokens.

### `path.py`

- Analytic straight-line interpolation path for flow matching.
- `build_flow_batch(e0, e1=None, t=None)`: samples or accepts `E1` and `t`, then builds `E0`, `E1`, `Et`, `velocity_target`, and `t`.
