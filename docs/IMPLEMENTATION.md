# AutoDec Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development if subagents are available, or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the revised `PROJECT.md` as a sibling `autodec/` package that reuses SuperDec where possible and adds the autoencoding path: pooled per-part residuals, SQ surface sampling, offset decoding, weighted reconstruction losses, training, evaluation, and tests.

**Architecture:** Keep the original `superdec/` package unchanged. Create `autodec/` at the same repository level as `superdec/`; start from the SuperDec package shape, but keep unchanged code as thin imports from `superdec` instead of maintaining a second full copy. Copy and modify only the files that must change: encoder feature exposure, heads with existence logits, residual projection, surface sampler, offset decoder, AE model wrapper, AE losses, training builders, and AE evaluation.

**Tech Stack:** PyTorch, Hydra/OmegaConf, existing SuperDec PVCNN/Transformer/data/sampling/loss utilities, `torch.cdist` for training Chamfer, existing KDTree evaluation utilities, pytest.

---

## Current Model Updates Beyond The Original Plan

The repository now includes several architectural updates that were not in the
first implementation plan:

- The residual projector pools assignment-weighted mean, masked max, and
  variance point-feature statistics before predicting `Z`. Its MLP input is
  `4H`, not the older `2H` mean-only input.
- The decoder keeps raw sampled SQ coordinates and concatenates Fourier
  features before projection.
- The decoder uses split projections for position/Fourier features, `E_dec`,
  residual `Z`, and the existence gate. Set `component_feature_dim=0` to recover
  the older raw concatenation path.
- The offset decoder is a stack of attention blocks: optional within-primitive
  self-attention, primitive cross-attention, and FFN, each with residual +
  LayerNorm.
- `lambda_cons` now uses a true no-residual decoder pass,
  `decoder(E_dec, Z=0)`, and remains disabled by default in YAML configs.
- Primitive-scale offset caps are implemented in `AutoDecDecoder`. The default
  YAML configs use `offset_cap: 0.4`; set `offset_cap: null` to restore the
  older unbounded offset behavior. Legacy scalar `offset_scale` still exists
  for compatibility when `offset_cap` is disabled.
- Training-time Z-dropout/residual dropout is not implemented. It remains a
  cheaper alternative or complement to the true second-pass consistency loss.

## What Changed In The Updated `PROJECT.md`

The newer spec changes the implementation plan in several concrete ways:

- Residual tokens are no longer a single linear projection from `F_SQ`. The
  current implementation uses multi-statistic assignment pooling:

  ```text
  g_mean,j = sum_i m_ij * F_PC,i / (sum_i m_ij + eps)
  g_max,j  = masked max_i m_ij * F_PC,i
  g_var,j  = sum_i m_ij * (F_PC,i - g_mean,j)^2 / (sum_i m_ij + eps)
  z_j      = MLP([F_SQ,j; g_mean,j; g_max,j; g_var,j])
  ```

  So the encoder must expose both point features `F_PC`, primitive features `F_SQ`, and assignment matrix `M`.

- Existence should be treated as a raw logit `a_j`, with:

  ```text
  alpha_j = sigmoid(a_j)
  w_j = sigmoid(a_j / tau)
  ```

  The current `superdec.models.heads.SuperDecHead` returns only `exist = sigmoid(exist_head(x))`. In `autodec`, copy/modify the head so it returns both `exist_logit` and `exist`.

- Decoder output is:

  ```text
  X_hat = X_sq + w * Delta
  ```

  The soft gate multiplies offsets, not SQ coordinates.

- Reconstruction loss is weighted Chamfer-L2. The forward term is weighted by `w`; the backward nearest-neighbor term must avoid matching target points to inactive scaffold points.

- The first prototype should omit normal alignment from `L_sq`, then add it only if needed for the paper-ready version.

## Feasibility

This is feasible in this repo. Most difficult infrastructure already exists in `superdec/`: the PVCNN encoder, transformer query decoder, SQ parameter heads, ShapeNet/ABO/ASE data loaders, equal-distance SQ sampler, pretrained checkpoint loading, LM code, and SQ fitting losses.

The easy parts are:

- Reusing `superdec.models.point_encoder.StackedPVConv`.
- Reusing `superdec.models.decoder.TransformerDecoder` and `superdec.models.decoder_layer.DecoderLayer`.
- Reusing `superdec.data.dataloader.ShapeNet`, `ABO`, and `ASE_Object`.
- Reusing `superdec.loss.sampler.EqualDistanceSamplerSQ`.
- Reusing `superdec.loss.utils.parametric_to_points_extended` for signed-power SQ sampling logic where appropriate.
- Reusing `superdec.loss.loss.SuperDecLoss` as the Phase 2 SQ regularizer, as long as `autodec` outdict keeps the same keys: `scale`, `shape`, `rotate`, `trans`, `exist`, and `assign_matrix`.

The difficult parts are:

- SuperDec's current forward discards `F_PC` and `F_SQ`; AutoDec needs them.
- `PROJECT.md` describes a 12-float explicit primitive code with 3 rotation parameters, but this repo's default head predicts a quaternion and returns a 3x3 rotation matrix. Keep this distinction explicit.
- Existing `superdec` losses and trainer code use `.cuda()` in places. If AutoDec has its own trainer/losses, keep them device-safe from the start.
- The equal-distance sampler chooses eta/omega via detached NumPy/C++ arrays. That is acceptable for v1 because points are still differentiable with respect to scale, shape, rotation, and translation, and it matches existing SuperDec behavior.
- Residual collapse is a real risk. The plan must monitor offset ratio, active primitive count, primitive assignment entropy, and SQ-only reconstruction quality.

## Important Repo-Specific Corrections

### Rotation And Bottleneck Size

`PROJECT.md` says `E` has 12 values: scale 3, shape 2, translation 3, rotation 3, existence 1. The current repo does not expose rotation this way:

- Default `superdec/models/heads.py` predicts a 4D quaternion internally, normalizes it, and returns a 3x3 matrix in `outdict["rotate"]`.
- Some configs use a 6D rotation head and still return a 3x3 matrix.

For v1:

- Use a decoder conditioning feature `E_dec` with 18 floats:

  ```text
  scale 3 + shape 2 + translation 3 + rotation matrix 9 + existence/logit 1 = 18
  ```

- Also keep a serialized/reporting code `E_ser`:

  ```text
  scale 3 + shape 2 + translation 3 + quaternion 4 + existence_logit 1 = 13
  ```

- If exact 12-float reporting is mandatory, add a serialization-only axis-angle conversion to 3 rotation values. Do not change the pretrained rotation head just to satisfy the table.

### Existence Logit Compatibility

The copied AutoDec head should expose:

```python
exist_logit = self.exist_head(x)
exist = torch.sigmoid(exist_logit)
```

Keep `exist` because existing SuperDec losses expect probabilities. Use `exist_logit` for temperature-gated decoder weights:

```python
gate = torch.sigmoid(exist_logit / tau)
```

When loading old checkpoints, this remains compatible because the `exist_head.weight` and `exist_head.bias` names do not change.

### Package Isolation

Do not modify `superdec/` for the AutoDec implementation unless a bug affects both projects. Put new or modified behavior under `autodec/`. Where behavior is identical, import from `superdec` explicitly so future diffs show what AutoDec owns.

## Target Tree

The tree below is the desired end state. Labels in comments mean:

- `new`: write from scratch.
- `copy+modify`: copy the SuperDec file, then change it.
- `re-export`: file only imports from `superdec` so AutoDec has a stable package shape without duplicating code.
- `config`: Hydra config.
- `test`: pytest file mirroring the `autodec/` package path.

```text
autodec/
  __init__.py                                  # new: exports AutoDec and AutoDecEncoder
  autodec.py                                   # new: full AE model wrapper
  encoder.py                                   # copy+modify from superdec/superdec.py

  models/
    __init__.py                                # new
    heads.py                                   # copy+modify from superdec/models/heads.py
    heads_mlp.py                               # optional copy+modify if using MLP heads
    heads_mlps.py                              # optional copy+modify if using MLPS heads
    point_encoder.py                           # re-export from superdec.models.point_encoder
    transformer_decoder.py                     # re-export TransformerDecoder from superdec.models.decoder
    decoder_layer.py                           # re-export DecoderLayer from superdec.models.decoder_layer
    residual.py                                # new: weighted part pooling + residual MLP
    offset_decoder.py                          # new: MLP and cross-attention offset decoders

  sampling/
    __init__.py                                # new
    sq_surface.py                              # new: SQSurfaceSampler using SuperDec sampler

  losses/
    __init__.py                                # new
    chamfer.py                                 # new: weighted Chamfer-L2
    autodec_loss.py                            # new: Phase 1/Phase 2 loss wrapper
    sq_regularizer.py                          # new: wrapper around superdec.loss.loss.SuperDecLoss

  data/
    __init__.py                                # re-export ShapeNet, ABO, ASE_Object, normalization helpers

  utils/
    __init__.py                                # new
    packing.py                                 # new: E_dec/E_ser packing and part repeats
    checkpoints.py                             # new: encoder checkpoint loading helpers
    metrics.py                                 # new: offset ratio, active primitive count, entropy
    correspondence.py                          # later: Hungarian matching for edits
    editing.py                                 # later: remove/scale/swap/interpolate primitives

  training/
    __init__.py                                # new
    builders.py                                # new: build model/loss/optimizer/dataloaders
    trainer.py                                 # new or copy+modify from train/trainer.py
    train.py                                   # new Hydra entrypoint for AutoDec

  evaluate/
    __init__.py                                # new
    evaluate.py                                # new: direct decoded-point evaluation
    to_npz.py                                  # later: save E/Z/decoded points

configs/
  autodec/
    train_phase1.yaml                          # config
    train_phase2.yaml                          # config
    eval.yaml                                  # config
    smoke.yaml                                 # config: tiny one-category smoke run

tests/
  autodec/
    test_autodec.py                            # test: high-level model forward
    test_encoder.py                            # test: feature exposure and checkpoint compatibility

    models/
      test_heads.py                            # test: exist_logit + exist compatibility
      test_residual.py                         # test: weighted pooling and residual shape
      test_offset_decoder.py                   # test: MLP/cross-attn decoder shapes

    sampling/
      test_sq_surface.py                       # test: sampler shape, gates, gradients

    losses/
      test_chamfer.py                          # test: weighted Chamfer behavior
      test_autodec_loss.py                     # test: phase-specific loss composition

    utils/
      test_packing.py                          # test: E_dec/E_ser packing
      test_metrics.py                          # test: offset ratio, entropy, active count
      test_checkpoints.py                      # test: load SuperDec state dict into AutoDecEncoder

    training/
      test_builders.py                         # test: config builds model/loss/optimizer
```

## Copy And Import Policy

Start from SuperDec, but avoid maintaining two full copies:

```bash
cp -R superdec autodec
```

Then immediately make these ownership decisions:

```text
autodec/encoder.py
  Own this file. It replaces copied autodec/superdec.py as the AutoDec encoder.
  It should import autodec.models.heads so existence logits are available.

autodec/models/heads.py
  Own this file. It is copied from superdec/models/heads.py and modified to return exist_logit.

autodec/models/heads_mlp.py and autodec/models/heads_mlps.py
  Own only if configs need these head types. Otherwise defer them.

autodec/models/point_encoder.py
  Re-export:
    from superdec.models.point_encoder import *

autodec/models/transformer_decoder.py
  Re-export:
    from superdec.models.decoder import TransformerDecoder

autodec/models/decoder_layer.py
  Re-export:
    from superdec.models.decoder_layer import DecoderLayer

autodec/data/__init__.py
  Re-export:
    from superdec.data.dataloader import ShapeNet, ABO, ASE_Object, normalize_points, denormalize_points, denormalize_outdict

autodec/sampling/sq_surface.py
  Import:
    from superdec.loss.sampler import EqualDistanceSamplerSQ

autodec/losses/sq_regularizer.py
  Import:
    from superdec.loss.loss import SuperDecLoss

autodec/utils/metrics.py
  Import evaluation helpers if needed:
    from superdec.utils.evaluation import get_outdict
```

Remove or ignore copied directories that AutoDec does not own:

```text
autodec/functional/          # do not duplicate; point_encoder re-export uses superdec.functional
autodec/fast_sampler/        # do not duplicate; SQ sampler imports superdec.fast_sampler through superdec.loss.sampler
autodec/lm_optimization/     # defer; LM is evaluation-only ablation
autodec/visualization/       # defer until qualitative demos
autodec/visualize_dataset.py # not needed for core AE
```

If keeping placeholder modules for import compatibility, make them explicit re-exports rather than copied code.

## File Responsibilities

### `autodec/encoder.py`

Copy `superdec/superdec.py`, rename class to `AutoDecEncoder`, and modify forward output.

Responsibilities:

- Build the same point encoder, transformer decoder, and SQ heads as SuperDec.
- Load pretrained SuperDec state dicts with unchanged submodule names.
- Return all original outdict keys plus:

  ```text
  point_features: [B, N, H]
  sq_features:    [B, P, H]
  exist_logit:    [B, P, 1]
  assign_matrix:  [B, N, P]
  ```

Implementation details:

```python
def forward(self, x, return_features=True):
    point_features = self.point_encoder(x)
    refined_queries_list, assign_logits = self.layers(self.init_queries, point_features)
    q = refined_queries_list[-1][:, :-1, :]
    out = self.heads(q)
    out["assign_matrix"] = torch.softmax(assign_logits[-1], dim=2)
    if return_features:
        out["point_features"] = point_features
        out["sq_features"] = q
    return out
```

Keep the existing optional LM path disabled for AutoDec v1.

### `autodec/models/heads.py`

Copy from `superdec/models/heads.py`.

Required changes:

- Return raw existence logits:

  ```python
  exist_logit = self.exist_head(x)
  exist = self.exist_activation(exist_logit)
  out_dict = {
      "scale": scale,
      "shape": shape,
      "rotate": rotation,
      "trans": translation,
      "exist_logit": exist_logit,
      "exist": exist,
  }
  ```

- Optionally return serialized rotation:

  ```python
  out_dict["rotation_quat"] = q
  ```

  for quaternion configs. This is useful for compression reporting, not for decoder conditioning.

Do not rename existing parameters. `exist_head.weight`, `exist_head.bias`, `scale_head.*`, etc. must stay checkpoint-compatible.

### `autodec/models/residual.py`

New file.

Responsibilities:

- Compute pooled local point evidence per primitive:

  ```python
  mass = assign_matrix.sum(dim=1).clamp_min(eps)       # [B, P]
  pooled = torch.einsum("bnp,bnh->bph", assign_matrix, point_features)
  mean = pooled / mass.unsqueeze(-1)
  max_features = masked_max(assign_matrix * point_features)
  var = sum_i assign_matrix * (point_features - mean)^2 / mass
  ```

- Concatenate pooled statistics with SQ query features:

  ```python
  x = torch.cat([sq_features, mean, max_features, var], dim=-1)  # [B, P, 4H]
  ```

- Apply MLP:

  ```python
  Linear(4H, H) -> ReLU -> Linear(H, d)
  ```

Class:

```python
class PartResidualProjector(nn.Module):
    def __init__(self, feature_dim=128, residual_dim=64, eps=1e-6):
        ...

    def forward(self, sq_features, point_features, assign_matrix):
        ...
```

### `autodec/utils/packing.py`

New file.

Responsibilities:

- Pack decoder conditioning features:

  ```python
  E_dec = [scale, shape, trans, rotate.flatten(-2), exist_logit or exist]
  # shape [B, P, 18]
  ```

- Pack serialized/reporting features:

  ```python
  E_ser = [scale, shape, trans, rotation_quat or rotation6d or axis_angle, exist_logit]
  ```

- Repeat per-primitive values to sampled surface points:

  ```python
  values[:, part_ids, :]
  ```

Functions:

```python
def pack_decoder_primitive_features(outdict) -> torch.Tensor: ...
def pack_serialized_primitive_features(outdict, rotation_mode="quat") -> torch.Tensor: ...
def repeat_by_part_ids(values, part_ids) -> torch.Tensor: ...
```

### `autodec/sampling/sq_surface.py`

New file.

Responsibilities:

- Use `EqualDistanceSamplerSQ` to sample `S_dec` eta/omega pairs per primitive.
- Convert eta/omega to canonical SQ surface points with signed powers.
- Transform canonical points to world points using `rotate` and `trans`.
- Compute gates from `exist_logit`:

  ```python
  weights = sigmoid(exist_logit / tau).repeat_interleave(S_dec, dim=1)
  ```

- Return a small dataclass or named tuple:

  ```text
  canonical_points: [B, P, S, 3]
  surface_points:   [B, P, S, 3]
  flat_points:      [B, P*S, 3]
  part_ids:         [P*S]
  weights:          [B, P*S]
  ```

Do not multiply coordinates by weights.

### `autodec/models/offset_decoder.py`

New file.

Responsibilities:

- Implement the current `CrossAttentionOffsetDecoder`:

  ```text
  point input dim     = 4 * component_feature_dim by default
  primitive token dim = 2 * component_feature_dim by default
  ```

- Preserve the legacy raw path when `component_feature_dim=0`:

  ```text
  point dim     = position_features + E_dec 18 + Z + gate 1
  primitive dim = E_dec 18 + Z
  ```

- Stack `n_blocks` attention blocks. Each block may run within-primitive
  self-attention on `[B * P, S, H]`, then cross-attention from sampled points to
  primitive tokens, then an FFN.

- Keep scalar `offset_scale` support for compatibility, but prefer the
  implemented primitive-scale caps:

  ```text
  offsets = tanh(raw_offsets) * offset_cap * mean(scale_{part(i)})
  ```

  where the default YAML value is `offset_cap: 0.4`.

Output convention:

```python
offsets = decoder(features)
decoded_points = surface_points + weights.unsqueeze(-1) * offsets
```

### `autodec/autodec.py`

New file.

Responsibilities:

- Own the full AE forward pass.
- Keep encoder and decoder separately addressable for freezing and optimizer groups.
- Return a rich outdict for training and evaluation.

Class shape:

```python
class AutoDec(nn.Module):
    def __init__(self, cfg):
        self.encoder = AutoDecEncoder(cfg.autodec.encoder)
        self.residual_projector = PartResidualProjector(...)
        self.surface_sampler = SQSurfaceSampler(...)
        self.offset_decoder = build_offset_decoder(...)

    def forward(self, points):
        enc = self.encoder(points, return_features=True)
        z = self.residual_projector(
            enc["sq_features"],
            enc["point_features"],
            enc["assign_matrix"],
        )
        sample = self.surface_sampler(enc)
        e_dec = pack_decoder_primitive_features(enc)
        features = decoder_feature_builder(sample.flat_points, E_by_point, Z_by_point, gate)
        offsets = self.offset_decoder(features)
        decoded = sample.flat_points + sample.weights.unsqueeze(-1) * offsets
        enc.update({...})
        return enc
```

Expected output additions:

```text
residual:          [B, P, d]
surface_points:    [B, P*S, 3]
decoded_offsets:   [B, P*S, 3]
decoded_weights:   [B, P*S]
decoded_points:    [B, P*S, 3]
consistency_decoded_points: [B, P*S, 3] only when return_consistency=True
part_ids:          [P*S]
E_dec:             [B, P, 18]
E_ser:             [B, P, 13] or documented alternative
```

### `autodec/losses/chamfer.py`

New file.

Implement weighted Chamfer-L2 aligned with the updated spec.

```python
def weighted_chamfer_l2(pred, target, weights, eps=1e-6, min_backward_weight=1e-3):
    d = torch.cdist(pred, target, p=2).pow(2)

    w = weights.clamp_min(eps)
    forward = (d.min(dim=2).values * w).sum(dim=1) / w.sum(dim=1)

    penalty = d / weights.clamp_min(min_backward_weight).unsqueeze(-1)
    backward = penalty.min(dim=1).values.mean(dim=1)

    return (forward + backward).mean()
```

The `min_backward_weight` clamp is necessary to avoid infinite values while still discouraging target points from matching inactive predictions.

### `autodec/losses/sq_regularizer.py`

New file.

Responsibilities:

- Wrap `superdec.loss.loss.SuperDecLoss`.
- Feed it AutoDec outdict fields with existing names.
- Keep Phase 2 implementation simple:

  ```python
  class SQRegularizer(nn.Module):
      def __init__(self, cfg):
          self.impl = SuperDecLoss(cfg)

      def forward(self, batch, outdict):
          return self.impl(batch, outdict)
  ```

If `SuperDecLoss` hardcodes CUDA and breaks CPU tests, do not patch `superdec/` casually. Either:

- make a device-safe wrapper in `autodec/losses/sq_regularizer.py`, or
- patch `superdec/loss/loss.py` only if the fix is general and low-risk.

### `autodec/losses/autodec_loss.py`

New file.

Responsibilities:

- Phase 1: `L = L_recon`.
- Phase 2: `L = L_recon + lambda_sq L_sq + lambda_par/L_exist through SQRegularizer`.
- Optional: add `L_cons` in either phase when `lambda_cons > 0`. It must use
  `consistency_decoded_points` from a second decoder pass with `Z=0`, not raw
  `surface_points`.
- Keep `scaffold_chamfer` as a no-grad diagnostic metric.
- Log:

  ```text
  recon
  sq_loss
  offset_ratio
  active_primitive_count
  primitive_mass_entropy
  scaffold_chamfer
  active_weight_sum
  ```

### `autodec/training/*`

New package instead of modifying root `train/`.

`autodec/training/builders.py`:

- Build `AutoDec`.
- Load pretrained SuperDec encoder checkpoint with:

  ```python
  model.encoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
  ```

- Build Phase 1 optimizer:

  ```text
  freeze encoder
  train residual_projector + offset_decoder
  lr = 1e-4
  ```

- Build Phase 2 optimizer:

  ```text
  encoder lr = 1e-5
  decoder/residual lr = 1e-4
  ```

`autodec/training/trainer.py`:

- Copy the useful parts of `train/trainer.py`.
- Make all batch movement device-safe with `.to(device)`.
- Save full AutoDec checkpoints.

`autodec/training/train.py`:

- Hydra entrypoint with config path `../../configs`.
- Run via:

  ```bash
  python -m autodec.training.train --config-name autodec/train_phase1
  ```

### `autodec/evaluate/evaluate.py`

New direct point-cloud evaluator.

Responsibilities:

- Load AutoDec checkpoint and config.
- Decode points directly.
- Prune points with `alpha > threshold` for reporting.
- If a fixed count is required, top-k or resample to 4096.
- Report:

  ```text
  mean_chamfer_l1
  mean_chamfer_l2
  mean_f_score
  avg_active_primitives
  avg_active_decoded_points
  avg_offset_ratio
  scaffold_chamfer_l2
  decoded_chamfer_l2
  ```

## Implementation Tasks

### Task 1: Create `autodec/` Package Skeleton

**Files:**

- Create: `autodec/__init__.py`
- Create: `autodec/autodec.py`
- Create: `autodec/encoder.py`
- Create directories listed in the target tree.
- Modify: `pyproject.toml`
- Test: `tests/autodec/test_autodec.py`

- [ ] Copy `superdec/` to `autodec/` as a starting point.
- [ ] Remove or replace unchanged copied files with re-export stubs.
- [ ] Update `pyproject.toml` package discovery:

```toml
[tool.setuptools.packages.find]
include = ["superdec*", "superoptim*", "superdec_planner*", "autodec*"]
```

- [ ] Add import smoke test:

```python
def test_import_autodec():
    import autodec
    assert autodec is not None
```

- [ ] Run:

```bash
python -m pytest tests/autodec/test_autodec.py -q
```

Expected: import succeeds.

### Task 2: Implement AutoDec Heads With Logits

**Files:**

- Modify: `autodec/models/heads.py`
- Optional modify: `autodec/models/heads_mlp.py`
- Optional modify: `autodec/models/heads_mlps.py`
- Test: `tests/autodec/models/test_heads.py`

- [ ] Copy `superdec/models/heads.py` into `autodec/models/heads.py`.
- [ ] Return `exist_logit` and `exist`.
- [ ] Keep parameter names unchanged for checkpoint compatibility.
- [ ] Add test:

```python
def test_head_returns_exist_logit_and_probability():
    head = SuperDecHead(emb_dims=128, ctx=ctx)
    out = head(torch.randn(2, 16, 128))
    assert out["exist_logit"].shape == (2, 16, 1)
    assert out["exist"].shape == (2, 16, 1)
    assert torch.allclose(out["exist"], torch.sigmoid(out["exist_logit"]))
```

### Task 3: Implement Feature-Exposing Encoder

**Files:**

- Modify: `autodec/encoder.py`
- Test: `tests/autodec/test_encoder.py`
- Test: `tests/autodec/utils/test_checkpoints.py`

- [ ] Copy `superdec/superdec.py` logic into `AutoDecEncoder`.
- [ ] Import AutoDec heads from `autodec.models.heads`.
- [ ] Return `point_features`, `sq_features`, `assign_matrix`, and all original SQ fields.
- [ ] Add checkpoint compatibility test using a fake SuperDec state dict or a real checkpoint if available.
- [ ] Ensure `AutoDecEncoder.load_state_dict(superdec_checkpoint["model_state_dict"], strict=True)` works when head type matches.

### Task 4: Implement Residual Projection

**Files:**

- Create: `autodec/models/residual.py`
- Test: `tests/autodec/models/test_residual.py`

- [ ] Implement weighted pooling from `assign_matrix` and `point_features`.
- [ ] Implement `MLP([F_SQ; pooled]) -> Z`.
- [ ] Test exact weighted pooling on a small hand-built tensor.
- [ ] Test output shape `[B, P, d]`.

### Task 5: Implement Primitive Packing

**Files:**

- Create: `autodec/utils/packing.py`
- Test: `tests/autodec/utils/test_packing.py`

- [ ] Implement `pack_decoder_primitive_features(outdict)` returning `[B, P, 18]`.
- [ ] Implement `pack_serialized_primitive_features(outdict)` returning a documented serialized shape.
- [ ] Implement `repeat_by_part_ids(values, part_ids)`.
- [ ] Test that `E_dec` uses matrix rotation and `E_ser` uses quaternion/declared mode.

### Task 6: Implement SQ Surface Sampler

**Files:**

- Create: `autodec/sampling/sq_surface.py`
- Test: `tests/autodec/sampling/test_sq_surface.py`

- [ ] Use `EqualDistanceSamplerSQ`.
- [ ] Compute signed-power surface points.
- [ ] Transform points with `rotate` and `trans`.
- [ ] Compute weights from `exist_logit / tau`.
- [ ] Test shapes:

```text
flat_points: [B, P*S, 3]
weights:     [B, P*S]
part_ids:    [P*S]
```

- [ ] Test gradients flow to `scale`, `shape`, `rotate`, and `trans` through point coordinates.

### Task 7: Implement Offset Decoder

**Files:**

- Create: `autodec/models/offset_decoder.py`
- Test: `tests/autodec/models/test_offset_decoder.py`

- [x] Implement stacked cross-attention decoder.
- [x] Add Fourier surface-point features in `AutoDecDecoder`.
- [x] Add split projections for position, E_dec, Z, and gate.
- [x] Add primitive-scale cap:

```text
offsets = tanh(raw_offsets) * cap * mean_scale_per_point
```

- [x] Return decoded offsets `[B, P*S, 3]` before existence gating.
- [x] Do offset gating in `AutoDec`, not inside the decoder.
- [x] Add cross-attention class with within-primitive self-attention blocks.

### Task 8: Implement Full `AutoDec`

**Files:**

- Modify: `autodec/autodec.py`
- Modify: `autodec/__init__.py`
- Test: `tests/autodec/test_autodec.py`

- [ ] Wire encoder, residual projector, sampler, packer, and decoder.
- [ ] Compute:

```python
decoded_points = surface_points + weights.unsqueeze(-1) * decoded_offsets
```

- [ ] Return all training keys.
- [x] Support `return_consistency=True`, which decodes the same sampled surface
  with `residual = 0`.
- [ ] Add `freeze_encoder()` and `unfreeze_encoder()`.
- [ ] Add `decoder_parameters()` yielding residual projector and offset decoder params.
- [ ] Test full forward shapes.

### Task 9: Implement Weighted Chamfer And AutoDec Loss

**Files:**

- Create: `autodec/losses/chamfer.py`
- Create: `autodec/losses/sq_regularizer.py`
- Create: `autodec/losses/autodec_loss.py`
- Test: `tests/autodec/losses/test_chamfer.py`
- Test: `tests/autodec/losses/test_autodec_loss.py`

- [ ] Implement weighted Chamfer-L2 from the updated spec.
- [ ] Wrap `SuperDecLoss` for Phase 2 SQ regularization.
- [ ] Implement Phase 1 and Phase 2 loss composition.
- [ ] Add metrics to loss dict.
- [ ] Test inactive prediction points do not dominate forward Chamfer.
- [ ] Test backward term discourages matching to low-weight inactive points.

### Task 10: Implement Metrics

**Files:**

- Create: `autodec/utils/metrics.py`
- Test: `tests/autodec/utils/test_metrics.py`

- [ ] Implement `offset_ratio(surface_points, offsets)`.
- [ ] Implement `active_primitive_count(exist, threshold=0.5)`.
- [ ] Implement `primitive_mass_entropy(assign_matrix)`.
- [ ] Implement `scaffold_vs_decoded_chamfer(surface_points, decoded_points, points, weights)`.

### Task 11: Implement AutoDec Training

**Files:**

- Create: `autodec/training/builders.py`
- Create: `autodec/training/trainer.py`
- Create: `autodec/training/train.py`
- Create: `configs/autodec/train_phase1.yaml`
- Create: `configs/autodec/train_phase2.yaml`
- Create: `configs/autodec/smoke.yaml`
- Test: `tests/autodec/training/test_builders.py`

- [ ] Build dataloaders by importing `ShapeNet`, `ABO`, and `ASE_Object` from `superdec.data.dataloader`.
- [ ] Build Phase 1 optimizer with frozen encoder.
- [ ] Build Phase 2 optimizer with differential learning rates.
- [ ] Load encoder-only checkpoint from SuperDec for Phase 1.
- [ ] Resume full AutoDec checkpoint for Phase 2.
- [ ] Keep scheduler disabled initially.
- [ ] Add smoke config for one category, small batch, no workers.

Run commands:

```bash
python -m autodec.training.train --config-name autodec/smoke
python -m autodec.training.train --config-name autodec/train_phase1
python -m autodec.training.train --config-name autodec/train_phase2
```

### Task 12: Implement AutoDec Evaluation

**Files:**

- Create: `autodec/evaluate/evaluate.py`
- Create: `configs/autodec/eval.yaml`

- [ ] Load AutoDec checkpoint.
- [ ] Decode points directly.
- [ ] Prune inactive primitive points for reporting.
- [ ] Resample/top-k to 4096 if needed for fair baseline comparisons.
- [ ] Report Chamfer-L1, Chamfer-L2, F-score, active primitive count, active decoded points, offset ratio, scaffold Chamfer, decoded Chamfer.

Run:

```bash
python -m autodec.evaluate.evaluate --config-name autodec/eval
```

### Task 13: Add Editing And Correspondence Later

**Files:**

- Create later: `autodec/utils/correspondence.py`
- Create later: `autodec/utils/editing.py`
- Tests later: `tests/autodec/utils/test_correspondence.py`
- Tests later: `tests/autodec/utils/test_editing.py`

- [ ] Implement Hungarian matching over active primitives.
- [ ] Implement remove, scale, swap, interpolate operations on `(E, Z)`.
- [ ] Restrict cross-object operations to same-category examples.
- [ ] Document primitive selection protocol.

## Config Sketches

### `configs/autodec/train_phase1.yaml`

```yaml
run_name: autodec_phase1_mlp_d64
seed: 42
device: cuda
use_wandb: false

model:
  name: autodec

checkpoints:
  encoder_from: checkpoints/shapenet/ckpt.pt
  resume_from: null
  keep_epoch: false

autodec:
  encoder:
    decoder:
      n_queries: 16
      n_layers: 3
      n_heads: 1
      masked_attention: false
      swapped_attention: false
      dim_feedforward: 512
      deep_supervision: false
      pos_encoding_type: sinusoidal
    point_encoder:
      in_channels: 3
      out_channels: 128
      kernel_size: 3
      resolution: 32
      # copy the nested l1/l2/l3/voxelization fields from configs/train.yaml
  residual_dim: 64
  n_surface_samples: 256
  exist_tau: 1.0
  decoder:
    hidden_dim: 128
    n_heads: 4
    positional_frequencies: 6
    component_feature_dim: null
    n_blocks: 2
    self_attention_mode: within_primitive
    offset_scale: null
    offset_cap: 0.4

dataset: shapenet
shapenet:
  path: data/ShapeNet
  categories: null
  normalize: false

trainer:
  save_path: checkpoints
  save_every_n_epochs: 50
  evaluate_every_n_epochs: 1
  num_epochs: 200
  batch_size: 32
  num_workers: 4
  augmentations: true

optimizer:
  decoder_lr: 1e-4
  encoder_lr: 1e-5
  weight_decay: 0
  betas: [0.9, 0.999]
  enable_scheduler: false

loss:
  type: autodec
  phase: 1
  lambda_sq: 0.0
  lambda_par: 0.0
  lambda_exist: 0.0
  lambda_cons: 0.0
  n_sq_samples: 256
  min_backward_weight: 1e-3
```

### `configs/autodec/train_phase2.yaml`

Copy Phase 1 and change:

```yaml
run_name: autodec_phase2_mlp_d64
checkpoints:
  encoder_from: null
  resume_from: checkpoints/autodec_phase1_mlp_d64/epoch_200.pt
  keep_epoch: false

loss:
  type: autodec
  phase: 2
  lambda_sq: 1.0
  lambda_par: 0.06
  lambda_exist: 0.01
  lambda_cons: 0.0
  n_sq_samples: 256
  min_backward_weight: 1e-3
```

## Verification Checklist

Before calling the implementation complete:

- [ ] `python -m pytest tests/autodec -q` passes.
- [ ] `python -m pip install -e .` includes `autodec`.
- [ ] `python setup_sampler.py build_ext --inplace` succeeds in the target environment.
- [ ] `python -m autodec.training.train --config-name autodec/smoke` runs without NaNs.
- [ ] `python -m autodec.training.train --config-name autodec/train_phase1 trainer.num_epochs=2 shapenet.categories='["03001627"]'` runs.
- [ ] `python -m autodec.training.train --config-name autodec/train_phase2 trainer.num_epochs=2 shapenet.categories='["03001627"]'` runs after a Phase 1 checkpoint exists.
- [ ] `python -m autodec.evaluate.evaluate --config-name autodec/eval` prints metrics from an AutoDec checkpoint.
- [ ] Original `superdec/` import still works:

```bash
python -c "from superdec.superdec import SuperDec; print('ok')"
```

- [ ] AutoDec import works:

```bash
python -c "from autodec import AutoDec; print('ok')"
```

- [ ] Phase 1 logs finite `recon`, `offset_ratio`, `active_weight_sum`.
- [ ] Phase 2 logs finite `recon`, `sq_loss`, `offset_ratio`, `active_primitive_count`, `primitive_mass_entropy`.
- [ ] Compression reporting distinguishes `E_dec` internal features from serialized `E_ser`.
- [ ] Shape completion claims are not made unless partial-input training or structured occlusion evaluation has been run.

## Recommended Order

1. Create `autodec/` skeleton and package discovery.
2. Implement `autodec/models/heads.py` with `exist_logit`.
3. Implement `autodec/encoder.py` and checkpoint compatibility.
4. Implement `autodec/models/residual.py`.
5. Implement `autodec/utils/packing.py`.
6. Implement `autodec/sampling/sq_surface.py`.
7. Implement `autodec/models/offset_decoder.py`.
8. Implement `autodec/autodec.py`.
9. Implement `autodec/losses/*`.
10. Implement `autodec/utils/metrics.py`.
11. Implement `autodec/training/*` and configs.
12. Run smoke Phase 1.
13. Run full Phase 1.
14. Run smoke Phase 2.
15. Implement `autodec/evaluate/evaluate.py`.
16. Only then decide whether to add cross-attention, consistency loss, or editing utilities.
