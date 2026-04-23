# gendec

## Purpose

`gendec/` is the generative scaffold package. It now owns both:

- Phase 1: unconditional scaffold-only generation over explicit SQ tokens `E`
- Phase 2: unconditional joint generation over AutoDec bottleneck tokens `(E, Z)`

It owns:

- teacher export from frozen SuperDec or AutoDec outputs into ordered scaffold or joint tokens
- normalized token datasets over ShapeNet-style directory layouts
- token-space flow models and losses for Phase 1 and Phase 2
- training, sampling, and evaluation entrypoints for both phases
- shell scripts for export, training, validation evaluation, and test evaluation
- optional eval-only coarse decoding of sampled scaffolds through a frozen AutoDec decoder with zero residual latents
- Phase 2 joint decoding of sampled `(E,Z)` tokens through the frozen AutoDec decoder, with the sampled existence threshold applied before coarse reconstruction export
- generated-SQ visualization exports for test evaluation under `data/viz/`, including Phase 2 `decoded_points.ply` reconstructions pruned to active primitives when frozen AutoDec decoding is enabled
- preview-video rendering under `gendec/videos/<run_name>/video_000000.mp4`, `video_000001.mp4`, and related per-sample outputs
- package-level documentation and tests

## Maintenance Contract

This file and every `{folder}.md` file under `gendec/` are part of the package contract.

When code or configs in a documented folder change, the matching folder markdown must be updated in the same change.

## Root Files

### `__init__.py`

- Re-exports the public package surface.
- Uses lazy attribute loading so lightweight entrypoints such as teacher export do not import the full eval/runtime stack at module import time.
- `ScaffoldTokenDataset`: dataset API for normalized scaffold tokens.
- `JointTokenDataset`: dataset API for normalized joint `(E,Z)` tokens.
- `Phase1Evaluator`, `Phase2Evaluator`: evaluator APIs for held-out token-space evaluation.
- `FlowMatchingLoss`, `JointFlowMatchingLoss`: Phase 1 and Phase 2 training losses.
- `SetTransformerFlowModel`, `JointSetTransformerFlowModel`: Phase 1 and Phase 2 neural networks.
- `postprocess_tokens`, `postprocess_joint_tokens`, `render_scaffold_preview`, `sample_scaffolds`, `sample_joint_scaffolds`: sampling-side helpers.

### `config.py`

- Small configuration utility layer shared by CLI entrypoints.
- `cfg_get(cfg, name, default=None)`: uniform getter for namespace or dict configs.
- `to_namespace(value)`: recursively converts dict/list config trees into `SimpleNamespace` objects.
- `load_yaml_config(path)`: loads YAML and returns namespace-style access.
- `fallback_cli_config(default_config_name)`: CLI fallback for environments without Hydra.
- `explicit_config_argument(default_config_name)`: lets entrypoints accept `--config path.yaml` even when Hydra is installed.

### `export_teacher.py`

- CLI entrypoint for dataset creation.
- `run_export(cfg)`: dispatches to Phase 1 toy export, Phase 2 toy export, or real teacher export based on `export.mode`.
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

### `train_phase2.py`

- CLI entrypoint for Phase 2 model training.
- `run_train_phase2(cfg)`: builds joint-token train/val dataloaders, the Phase 2 model/loss, optimizer, scheduler, optional WandB run, optionally restores a prior checkpoint, and runs `Phase2Trainer`.
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
- [`utils/utils.md`](./utils/utils.md): small shared runtime helpers such as console logging.
