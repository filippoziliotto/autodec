# eval

## Purpose

`gendec/eval/` contains the held-out evaluation runtime for Phase 1 and Phase 2 checkpoints. It evaluates token-space prediction quality on exported splits and also writes unconditional generation artifacts for inspection.

## Maintenance Contract

If evaluation metrics, outputs, or checkpoint-loading behavior change, this file must be updated in the same change.

## Files

### `__init__.py`

- Re-export surface for evaluation utilities.
- Exposes `Phase1Evaluator`, `Phase2Evaluator`, `MetricAverager`, `build_frozen_autodec_decoder`, `decode_scaffolds_with_zero_residual`, `decode_joint_scaffolds`, and `nearest_neighbor_paper_metrics`.

### `autodec_bridge.py`

- Eval-only bridge between sampled `gendec` tokens and a frozen AutoDec decoder.
- `build_frozen_autodec_decoder(config_path, checkpoint_path=None, device="cpu")`: loads a decoder config, restores only decoder weights from a checkpoint, freezes parameters, and returns the decoder plus residual dimension.
- `sampled_scaffolds_to_decoder_outdict(processed, residual_dim)`: converts sampled Phase 1 scaffolds into the decoder outdict with zero residuals.
- `sampled_joint_scaffolds_to_decoder_outdict(processed)`: converts sampled Phase 2 joint tokens into the decoder outdict using generated residuals and replaces soft existence with the sampled hard `active_mask`, encoded as binary `exist` plus `exist_logit in {-20, +20}` so the frozen decoder sees the same active/inactive primitive decision as the SQ visualizations.
- `decode_scaffolds_with_zero_residual(...)`: Phase 1 zero-residual decode.
- `decode_joint_scaffolds(...)`: Phase 2 decode using generated `Z`.

### `evaluator.py`

- Core evaluation loop implementation.
- `_batch_size(batch)`: extracts Phase 1 batch size from `tokens_e`.
- `_conditioned_generation_plan(cfg, dataset, model, split, device, requested_num_samples=None)`: resolves whether generated sampling should be unconditional or per-class, and returns the matching category-index batch plus category labels.
- `Phase1Evaluator`: held-out Phase 1 evaluation plus optional zero-residual AutoDec decode and generated-SQ visualization export.
  - When class conditioning is enabled and the exported dataset has more than one class, the Phase 1 test evaluator generates `eval.generated_per_class` samples for every category instead of a single global unconditional batch.
- `Phase2Evaluator`: held-out Phase 2 evaluation over `tokens_ez`, optional generated `(E,Z)` AutoDec decoding, and generated-SQ visualization export from the explicit scaffold portion.
  - When the frozen AutoDec decode branch is enabled on the Phase 2 test split, the evaluator prunes generated `surface_points` and `decoded_points` down to active primitives only before reporting coarse plausibility metrics, writing `generated_autodec_samples.pt`, and exporting `decoded_points.ply`.
  - The saved batch artifact now contains both pruned and raw point clouds: `decoded_points`, `surface_points`, `decoded_points_raw`, and `surface_points_raw`.
  - When class conditioning is enabled and the exported dataset has more than one class, the Phase 2 test evaluator generates `eval.generated_per_class` samples for every category and writes category-grouped visualization folders.

### `metrics.py`

- Metric aggregation and nearest-neighbor plausibility helpers used by evaluation.

### `run.py`

- CLI entrypoint for Phase 1 evaluation.
- `run_eval(cfg)`: builds the Phase 1 model, restores the configured checkpoint, builds the Phase 1 dataset/loss, and runs `Phase1Evaluator`.

### `run_phase2.py`

- CLI entrypoint for Phase 2 evaluation.
- `run_eval_phase2(cfg)`: builds the Phase 2 model, restores the configured checkpoint, builds the Phase 2 dataset/loss, and runs `Phase2Evaluator`.
