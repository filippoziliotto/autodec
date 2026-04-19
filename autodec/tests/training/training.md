# training

This folder tests `autodec/training/`.

Files:

```text
test_builders.py
test_trainer.py
```

## `test_builders.py`

Tests:

```text
autodec.training.builders.build_loss
autodec.training.builders.build_optimizer
autodec.training.builders.limit_dataset
```

The tests use `TinyAutoDec`, a minimal model with the same parameter-group
surface as the real `AutoDec` wrapper:

```text
encoder.point_encoder
encoder.layers
encoder.heads
encoder.residual_projector
decoder
```

### `test_build_loss_constructs_autodec_loss_from_cfg`

Builds a simple config:

```text
loss.phase = 2
loss.lambda_sq = 2
loss.lambda_par = 3
loss.lambda_exist = 4
loss.lambda_cons = 0
loss.n_sq_samples = 8
```

Checks:

```text
build_loss returns AutoDecLoss
loss.phase == 2
loss.lambda_sq == 2
```

Purpose:

Verify AutoDec loss can be constructed from config without using
`superdec.loss.loss.Loss`.

### `test_build_optimizer_phase1_trains_residual_and_decoder_only`

Uses:

```text
phase = 1
```

Checks:

```text
encoder.point_encoder is frozen
optimizer params == residual_projector params + decoder params
```

Purpose:

Protect decoder-warmup behavior.

### `test_build_optimizer_phase2_uses_differential_learning_rates`

Uses:

```text
phase = 2
encoder_lr = 1e-4
residual_lr = 3e-3
decoder_lr = 2e-3
```

Checks optimizer learning rates:

```text
[1e-4, 3e-3, 2e-3]
```

and confirms all parameters are trainable.

Purpose:

Protect phase-2 joint fine-tuning behavior.

### `test_limit_dataset_returns_deterministic_subset_when_limit_is_set`

Builds a list dataset with ten items and applies:

```text
max_items = 4
seed = 7
```

Checks:

```text
result is torch.utils.data.Subset
subset length is 4
same seed gives same indices
different seed gives different indices
indices are shuffled, not just the first items
```

Purpose:

Protect deterministic reduced-ShapeNet debugging runs.

### `test_limit_dataset_leaves_dataset_unchanged_when_limit_is_null_or_large`

Checks:

```text
max_items = null       -> original dataset object
max_items >= len(data) -> original dataset object
```

Purpose:

Ensure full training remains the default when no reduction is requested.

## `test_trainer.py`

Tests:

```text
autodec.training.trainer.AutoDecTrainer
```

### Local `TinyModel`

Has one trainable scalar parameter and returns:

```text
decoded_points
decoded_weights
```

from `forward(points)`.

### Local `TinyLoss`

Computes:

```text
decoded_points.square().mean()
```

and returns:

```text
(loss, {"all": loss_value})
```

### `test_autodec_trainer_runs_one_train_and_eval_epoch`

Builds one-batch train and validation loaders from lists of dictionaries:

```text
{"points": torch.ones(2, 3, 3)}
```

Runs:

```text
trainer.train_one_epoch(0)
trainer.evaluate(0)
```

Checks both returned metric dictionaries contain:

```text
"all"
```

Purpose:

Verify the trainer can run a complete train/eval step on CPU without hardcoded
`.cuda()` calls.

### `test_autodec_trainer_requests_consistency_pass_only_when_loss_needs_it`

Uses a tiny loss with:

```text
lambda_cons = 1.0
```

and a model that records the `return_consistency` flag passed to `forward`.

Checks:

```text
return_consistency == True
consistency_decoded_points is present before loss computation
```

Purpose:

Verify the trainer requests the extra zero-residual decoder pass only when the
configured loss needs `L_cons`.
