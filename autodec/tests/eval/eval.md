# autodec/tests/eval

Tests for the standalone AutoDec test-evaluation path.

`test_selectors.py` checks deterministic category-balanced sample selection from
ShapeNet-style `dataset.models` metadata.

`test_eval_metrics.py` checks paper-style Chamfer metric names and metric averaging.

`test_evaluator.py` uses tiny in-memory fakes to verify that the evaluator writes
summary metrics, per-sample metrics, and category-aware visualization metadata
without requiring ShapeNet files or a real AutoDec checkpoint.
