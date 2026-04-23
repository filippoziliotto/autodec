# eval

## Purpose

`gendec/tests/eval/` contains evaluation-specific tests for metric aggregation and the standalone evaluator runtime.

## Maintenance Contract

If evaluation test coverage changes in this subfolder, this file must be updated in the same change.

## Files

### `test_autodec_bridge.py`

- `_write_autodec_decoder_assets(tmp_path)`: writes a tiny AutoDec decoder config and decoder-only checkpoint for local bridge tests.
- `test_autodec_bridge_decodes_sampled_scaffolds_with_zero_residual`: verifies sampled `gendec` scaffolds can be converted and decoded through a frozen AutoDec decoder with zero residuals.
- `test_phase2_autodec_bridge_hard_thresholds_existence_inputs`: verifies the Phase 2 frozen-decode bridge replaces soft existence with the sampled hard `active_mask` before calling the decoder.
- `test_prune_points_by_active_primitives_filters_inactive_parts`: verifies active-primitive pruning removes decoded points that belong to inactive primitive ids.

### `test_evaluator.py`

- `_cfg(root, checkpoint_path, output_dir)`: builds a minimal namespace config for evaluator tests.
- `_cfg_phase2(root, checkpoint_path, output_dir, residual_dim=4, autodec_decode=None)`: builds a minimal namespace config for Phase 2 evaluator tests.
- `test_phase1_evaluator_writes_metrics_and_per_sample_rows`: verifies that `Phase1Evaluator` writes `metrics.json`, `per_sample_metrics.jsonl`, includes generated-sample summary metrics, and writes generated SQ visualization folders under the configured `data/viz/...` root.
- `test_phase1_evaluator_can_decode_generated_scaffolds_with_frozen_autodec_decoder`: verifies the optional frozen AutoDec decode branch writes coarse plausibility metrics and a decoded sample artifact.
- `test_phase2_evaluator_writes_decoded_point_cloud_visualizations`: verifies that `Phase2Evaluator` writes `decoded_points.ply` into each generated visualization folder when the frozen AutoDec decode branch is enabled, using the active-primitive-pruned Phase 2 reconstruction.
- `test_phase1_conditioned_test_eval_generates_five_samples_per_class`: verifies that conditioned Phase 1 test evaluation writes exactly five generated samples for each class into per-category visualization folders.
- `test_phase2_conditioned_test_eval_generates_five_samples_per_class`: verifies that conditioned Phase 2 test evaluation writes exactly five generated samples for each class into per-category visualization folders.

### `test_metrics.py`

- `test_metric_averager_computes_weighted_means`: verifies that `MetricAverager` computes proper weighted means across multiple updates.
- `test_nearest_neighbor_paper_metrics_selects_best_reference`: verifies that nearest-neighbor plausibility metrics choose the best available reference point cloud.
