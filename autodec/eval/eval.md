# autodec/eval

Standalone evaluation code for trained AutoDec checkpoints.

`run.py` is the Hydra entrypoint used by `autodec/scripts/run_eval_test.sh`.
It builds the real AutoDec model, loads a checkpoint, constructs the ShapeNet
`test` split, and launches `AutoDecEvaluator`.

`evaluator.py` contains the core evaluation loop. It runs the model in
`torch.no_grad()`, aggregates AutoDec loss metrics, aggregates paper-style
Chamfer metrics, writes `metrics.json`, writes `per_sample_metrics.jsonl`, and
produces category-balanced 3D visualization artifacts.

`selectors.py` contains deterministic ShapeNet sample selection. It reads
`dataset.models`, groups entries by category, requires the configured minimum
number of categories, and returns fixed dataset indices for visualization.

`metrics.py` contains the evaluation-only metrics. `paper_chamfer_metrics`
computes the symmetric Chamfer-L1 and Chamfer-L2 values used for paper-style
reporting, while `MetricAverager` accumulates weighted means for metric
dictionaries whose keys may differ between updates.
