# autodec/tests/eval

Tests for the standalone AutoDec test-evaluation path.

`test_selectors.py` checks deterministic sample selection from ShapeNet-style
`dataset.models` metadata, using every category and a fixed number of examples
per category.

`test_eval_metrics.py` checks paper-style Chamfer, x100-scaled Chamfer,
F-score metric names, configurable F-score thresholds, and metric averaging.

`test_evaluator.py` uses tiny in-memory fakes to verify that the evaluator writes
summary metrics, per-sample metrics, category-aware visualization metadata, and
pruned paper-metric/visualization inputs without requiring ShapeNet files or a
real AutoDec checkpoint. It also checks that loss evaluation requests
`return_consistency=True` when `lambda_cons > 0`, so the intended no-residual
consistency loss has the decoder output it needs.
