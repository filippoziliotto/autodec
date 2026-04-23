# tests

## Purpose

`gendec/tests/` contains the package-level unit and smoke tests for data contracts, geometry helpers, losses, checkpoints, documentation presence, and end-to-end toy execution across both phases.

## Maintenance Contract

If a new top-level test file is added here, or an existing test changes scope, this index must be updated in the same change.

## Files

### `test_checkpoints.py`

- Verifies checkpoint round-tripping restores model weights plus saved epoch/loss metadata.

### `test_console_logger.py`

- Verifies console metric formatting and end-of-epoch printed summaries for both Phase 1 and Phase 2 training.

### `test_dataset.py`

- Covers the Phase 1 exported-example schema, split manifests, normalization stats, and toy multi-split export path.

### `test_docs.py`

- Verifies the required folder-level markdown files exist for the maintained `gendec/` source directories.

### `test_flow_matching.py`

- Verifies the analytic flow path and the Phase 1 loss API.

### `test_imports.py`

- Verifies the package root remains lazily importable without eagerly constructing the eval/runtime dependency surface.

### `test_model.py`

- Verifies the Phase 1 Set Transformer flow model shape contract.

### `test_ordering.py`

- Verifies deterministic primitive ordering and aligned teacher-tensor reordering.

### `test_phase2_builders.py`

- Verifies the Phase 2 dataset, dataloader, model, loss, and train/val builder surface.

### `test_phase2_dataset.py`

- Verifies the Phase 2 `JointTokenDataset` schema, normalization, residual-dimension inference, and batching behavior.

### `test_phase2_loss.py`

- Verifies the split explicit/residual Phase 2 loss, including the optional existence auxiliary and per-sample metric return path.

### `test_phase2_model.py`

- Verifies the Phase 2 joint model output contract and differentiability.

### `test_phase2_sampling.py`

- Verifies joint Euler sampling, joint postprocessing, and joint scaffold sampling outputs.

### `test_phase2_smoke.py`

- End-to-end Phase 2 smoke test covering toy export, one-epoch training, checkpointing, preview writing, and held-out evaluation.

### `test_phase2_tokens.py`

- Verifies the Phase 2 token helpers, including concatenation, splitting, custom residual widths, and the default 79D token width.

### `test_preview_video.py`

- Verifies preview discovery and per-sample MP4 writing under `gendec/videos/<run_name>/`.

### `test_rotation.py`

- Verifies the teacher-side 6D rotation conversion and its inverse.

### `test_scripts.py`

- Verifies the checked-in Phase 1 and Phase 2 export/train/eval shell scripts are present in `gendec/scripts/`.

### `test_shapenet_index.py`

- Verifies exported split manifests and source ShapeNet scanning behavior.

### `test_smoke.py`

- End-to-end Phase 1 smoke test covering toy export, one-step training, checkpointing, preview writing, and sampling.

### `test_wandb.py`

- Verifies lazy WandB initialization and per-epoch train/val/sample metric logging for both Phase 1 and Phase 2 training.

## Subfolder

- [`eval/eval.md`](./eval/eval.md): evaluation-specific tests, including the frozen AutoDec coarse-decode bridge and generated-SQ visualization artifacts.
