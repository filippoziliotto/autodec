# eval

## Purpose

`gendec/eval/` contains the held-out evaluation runtime for Phase 1 checkpoints. It evaluates token-space prediction quality on exported test splits and also writes unconditional generation artifacts for inspection.

## Maintenance Contract

If evaluation metrics, outputs, or checkpoint-loading behavior change, this file must be updated in the same change.

## Files

### `__init__.py`

- Re-export surface for evaluation utilities.
- `Phase1Evaluator`: batch evaluation runtime.
- `MetricAverager`: weighted metric aggregation helper.
- `build_frozen_autodec_decoder`, `decode_scaffolds_with_zero_residual`: frozen AutoDec bridge helpers.
- `nearest_neighbor_paper_metrics`: generation-side nearest-neighbor plausibility metric.

### `autodec_bridge.py`

- Eval-only bridge between sampled `gendec` scaffolds and a frozen AutoDec decoder.
- `_lazy_autodec_decoder()`: lazy-imports the AutoDec decoder class plus checkpoint helpers.
- `_FallbackAngleSampler`: deterministic fallback sampler used when the compiled SuperDec fast sampler is unavailable.
- `_supports_equal_distance_sampler()`: checks whether the compiled equal-distance sampler is available.
- `_decoder_kwargs(autodec_cfg)`: translates an AutoDec config into `AutoDecDecoder` constructor kwargs.
- `build_frozen_autodec_decoder(config_path, checkpoint_path=None, device="cpu")`: loads an AutoDec decoder config, restores only decoder weights from a checkpoint, freezes parameters, and returns the decoder plus residual dimension.
- `sampled_scaffolds_to_decoder_outdict(processed, residual_dim)`: converts sampled scaffold fields into the outdict expected by `AutoDecDecoder`, with zero residual latents.
- `decode_scaffolds_with_zero_residual(processed, decoder, residual_dim, return_attention=False)`: runs the frozen decoder on sampled scaffolds using `Z=0`.

### `evaluator.py`

- Core evaluation loop implementation.
- `_batch_size(batch)`: extracts batch size from the normalized token tensor.
- `Phase1Evaluator`:
  - `__init__(cfg, model, loss_fn, dataset, device=None)`: stores runtime objects, resolves the split name, and builds the output directory.
  - `_loader()`: builds a deterministic evaluation dataloader.
  - `_move_batch(batch)`: moves tensor fields onto the evaluation device.
  - `_write_json(path, payload)`: writes formatted JSON artifacts.
  - `_write_jsonl(path, rows)`: writes per-sample JSONL rows.
  - `_autodec_decode_enabled()`: reports whether the frozen AutoDec eval branch is enabled.
  - `_get_frozen_autodec_decoder()`: lazily builds and caches the frozen AutoDec decoder bridge.
  - `evaluate()`: runs held-out loss evaluation, aggregates metrics, writes `metrics.json`, writes per-sample rows, writes `generated_samples.pt` from unconditional sampling, writes 10 generated SQ visualization folders under `data/viz/<run_name>/test/` when test visualization is enabled, and optionally writes `generated_autodec_samples.pt` plus coarse plausibility metrics from the frozen AutoDec decoder.
  - Evaluation-time unconditional generation now prefers `sampling.eval_steps` and `sampling.exist_threshold` when explicit `eval` overrides are absent, so training, sampling, and evaluation can share one sampling policy.

### `metrics.py`

- Small metric helpers used during evaluation.
- `_to_float(value)`: converts tensors or scalars into Python floats.
- `_threshold_key(value)`: formats F-score thresholds into stable metric-key suffixes.
- `_subsample_points(points, point_count)`: downsamples or tiles point clouds to a fixed count for bounded metric cost.
- `_paper_metrics_single(pred, target, f_score_threshold=0.01, eps=1e-8)`: computes symmetric Chamfer and F-score metrics for one prediction-target pair.
- `MetricAverager`:
  - `__init__()`: initializes weighted-sum and count stores.
  - `update(metrics, batch_size=1)`: accumulates weighted metric values.
  - `compute()`: returns weighted means for all tracked metrics.
- `active_primitive_count(exist, threshold=0.5)`: mean number of active primitives across a batch.
- `active_decoded_point_count(weights, threshold=0.5)`: mean number of active decoded points across a batch.
- `token_channel_mean_abs(tokens)`: mean absolute token magnitude.
- `nearest_neighbor_paper_metrics(pred, reference, prefix, point_count=None, f_score_threshold=0.01)`: compares each generated point cloud to the nearest reference point cloud and reports average nearest-neighbor paper-style metrics.

### `run.py`

- CLI entrypoint for evaluation.
- `run_eval(cfg)`: builds the model, restores the configured or validation-best checkpoint weights, builds dataset and loss, and runs `Phase1Evaluator`.
- `_main(cfg)`: prints the final aggregated metrics.
