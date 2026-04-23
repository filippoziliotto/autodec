# gendec

## Purpose

`gendec/` is the Phase 1 generative scaffold package. It owns:

- teacher export from SuperDec outputs into ordered scaffold tokens
- normalized token datasets over ShapeNet-style directory layouts
- the token-space flow model and loss
- training, sampling, and evaluation entrypoints
- shell scripts for export, training, validation evaluation, and test evaluation
- optional eval-only coarse decoding of sampled scaffolds through a frozen AutoDec decoder with zero residual latents
- package-level documentation and tests

## Maintenance Contract

This file and every `{folder}.md` file under `gendec/` are part of the package contract.

When code or configs in a documented folder change, the matching folder markdown must be updated in the same change.

## Root Files

### `__init__.py`

- Re-exports the public package surface.
- `ScaffoldTokenDataset`: dataset API for normalized scaffold tokens.
- `Phase1Evaluator`: evaluator API for held-out token-space evaluation.
- `FlowMatchingLoss`: main training loss.
- `SetTransformerFlowModel`: Phase 1 neural network.
- `postprocess_tokens`, `render_scaffold_preview`, `sample_scaffolds`: sampling-side helpers.

### `config.py`

- Small configuration utility layer shared by CLI entrypoints.
- `cfg_get(cfg, name, default=None)`: uniform getter for namespace or dict configs.
- `to_namespace(value)`: recursively converts dict/list config trees into `SimpleNamespace` objects.
- `load_yaml_config(path)`: loads YAML and returns namespace-style access.
- `fallback_cli_config(default_config_name)`: CLI fallback for environments without Hydra.

### `export_teacher.py`

- CLI entrypoint for dataset creation.
- `run_export(cfg)`: dispatches to toy export or real SuperDec teacher export based on `export.mode`, and can materialize `train`/`val`/`test` in one run.
- `_main(cfg)`: prints the export result for Hydra or fallback CLI execution.

### `sample.py`

- CLI entrypoint for unconditional scaffold sampling from a checkpoint.
- `run_sample(cfg)`: loads normalization stats, restores the best or explicit checkpoint, samples scaffolds, and saves `samples.pt`.
- `_main(cfg)`: prints the output artifact path.

### `sampling.py`

- Runtime sampling and postprocessing helpers used by sampling and evaluation.
- `euler_sample(...)`: integrates the learned velocity field from `t=1` to `t=0` in normalized token space.
- `postprocess_tokens(...)`: unnormalizes tokens, splits channels, clamps visualization-sensitive fields, converts rotations, and computes active masks.
- `_signed_power(...)`: signed exponent helper used for superquadric surface evaluation.
- `render_scaffold_preview(...)`: renders active superquadric preview points.
- `sample_scaffolds(...)`: end-to-end wrapper returning processed sampled scaffolds plus preview points.

### `tokens.py`

- Canonical token layout definition for the whole package.
- `PRIMITIVE_COUNT`: fixed slot count, currently `16`.
- `TOKEN_DIM`: token width, currently `15`.
- `SCALE_SLICE`, `SHAPE_SLICE`, `TRANS_SLICE`, `ROT6D_SLICE`, `EXIST_LOGIT_SLICE`: canonical channel slices.
- `build_scaffold_tokens(...)`: concatenates semantic channels into `[P, 15]` token tensors.
- `split_scaffold_tokens(tokens)`: splits a token tensor back into semantic fields.

### `train.py`

- CLI entrypoint for Phase 1 model training.
- `run_train(cfg)`: builds train/val dataloaders, model, loss, optimizer, scheduler, optional WandB run, trainer, then runs training with validation and sample-quality logging.
- `_main(cfg)`: prints the train result payload.

## Subfolders

- [`configs/configs.md`](./configs/configs.md): runtime configurations.
- [`data/data.md`](./data/data.md): teacher export, dataset indexing, normalization, and toy data generation.
- [`docs/docs.md`](./docs/docs.md): design and implementation writeups.
- [`eval/eval.md`](./eval/eval.md): evaluation runtime, plausibility metrics, and frozen AutoDec decode bridge.
- [`losses/losses.md`](./losses/losses.md): flow-path and training objectives.
- [`models/models.md`](./models/models.md): network architecture and geometry helpers.
- [`scripts/scripts.md`](./scripts/scripts.md): checked-in shell scripts for export, training, and eval.
- [`tests/tests.md`](./tests/tests.md): test coverage overview.
- [`training/training.md`](./training/training.md): builders, checkpoints, logging, and trainer loop.
