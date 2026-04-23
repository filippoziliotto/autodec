# scripts

## Purpose

`gendec/scripts/` contains the checked-in shell entrypoints for the standard Phase 1 and Phase 2 workflows.

## Maintenance Contract

If a script is added, removed, or its invocation contract changes, this file must be updated in the same change.

## Files

### `common.sh`

- Shared script bootstrap.
- Resolves the script directory, resolves the repo root, and `cd`s to the repository before the concrete command runs.

### `export_shapenet_teacher.sh`

- Standard Phase 1 dataset-extraction script.
- Runs `python gendec/export_teacher.py --config gendec/configs/teacher_export.yaml`.

### `export_shapenet_phase2_teacher.sh`

- Standard Phase 2 dataset-extraction script.
- Runs `python gendec/export_teacher.py --config gendec/configs/phase2_export.yaml`.

### `train_phase1.sh`

- Standard Phase 1 training script.
- Runs `python gendec/train.py --config gendec/configs/train.yaml`.
- If WandB logging is enabled, source `autodec/keys/keys.sh` or otherwise export `WANDB_API_KEY` before running it.

### `train_phase2.sh`

- Standard Phase 2 training script.
- Runs `python gendec/train_phase2.py --config gendec/configs/train_phase2.yaml`.
- If WandB logging is enabled, source `autodec/keys/keys.sh` or otherwise export `WANDB_API_KEY` before running it.

### `eval_val.sh`

- Phase 1 validation evaluation script.
- Runs `python gendec/eval/run.py --config gendec/configs/eval_val.yaml`.

### `eval_test.sh`

- Phase 1 held-out test evaluation script.
- Runs `python gendec/eval/run.py --config gendec/configs/eval.yaml`.

### `eval_phase2_val.sh`

- Phase 2 validation evaluation script.
- Runs `python gendec/eval/run_phase2.py --config gendec/configs/eval_phase2_val.yaml`.

### `eval_phase2_test.sh`

- Phase 2 held-out test evaluation script.
- Runs `python gendec/eval/run_phase2.py --config gendec/configs/eval_phase2.yaml`.
