# autodec/eval

Standalone evaluation code for trained AutoDec checkpoints.

`run.py` is the Hydra entrypoint used by `autodec/scripts/run_eval_test.sh`.
It builds the real AutoDec model, loads a checkpoint, constructs the ShapeNet
`test` split, and launches `AutoDecEvaluator`.

For ShapeNet, `run.py` resolves `shapenet.category_split` before constructing
the dataset. Use `all` for the in-category test set, `paper_unseen` for the
SuperDec out-of-category held-out classes, and `null` to preserve an explicit
`shapenet.categories` list.

`evaluator.py` contains the core evaluation loop. It runs the model in
`torch.no_grad()`, aggregates AutoDec loss metrics, aggregates paper-style point
cloud metrics, writes `metrics.json`, writes `per_sample_metrics.jsonl`, and
produces category-balanced 3D visualization artifacts.

When `eval.prune_decoded_points` is true, paper-style point-cloud metrics and
test visualizations use decoded points pruned by primitive existence. Training
loss metrics still use the raw fixed-size decoded tensor so they stay aligned
with the training objective.

If loss metrics are enabled and `lambda_cons > 0`, `evaluator.py` requests
`return_consistency=True` from the model. This computes the no-residual decoder
pass needed for the consistency loss. The default configs keep
`lambda_cons: 0.0`, so standard evaluation still runs a single decoder pass.

When `eval.use_lm_optimization` is true, `run.py` enables SuperDec LM
optimization on `model.encoder` after loading the AutoDec checkpoint and before
running evaluation. This is off by default and should be reported separately
from the main AutoDec result because the decoder receives refined SQ parameters
with the original residual latent. The current LM implementation requires CUDA.

For qualitative test-set outputs, `visualization.write_lm_optimized_sq_mesh`
can write an additional `sq_mesh_lm.obj` per selected sample. This path keeps
the normal evaluation behavior intact: `sq_mesh.obj`, reconstruction points,
and metrics are produced from the unrefined model output, while only the extra
mesh is generated from a cloned LM-refined SQ outdict. The LM mesh is saved on
disk but not logged to WandB; WandB logs only the normal SQ visualization. The
option is ignored outside `split: test`.

The two LM modes are mutually exclusive. Use
`eval.use_lm_optimization: false` with
`visualization.write_lm_optimized_sq_mesh: true` when you want the normal
forward pass to remain `original SQ + Z -> reconstruction` and only add a
separate `original SQ + LM` mesh. Set `eval.use_lm_optimization: true` only for
a separate full-forward LM ablation.

`selectors.py` contains deterministic ShapeNet sample selection. It reads
`dataset.models`, groups entries by category, and returns
`visualization.samples_per_category` fixed dataset indices from every available
test category. The default is two examples per category.

`metrics.py` contains the evaluation-only metrics. `paper_chamfer_metrics`
computes symmetric Chamfer-L1, symmetric Chamfer-L2, their x100 table-reporting
variants, and precision/recall/F-score at the configured threshold
`eval.f_score_threshold` (default `0.01`). `MetricAverager` accumulates
weighted means for metric dictionaries whose keys may differ between updates.
