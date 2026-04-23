# scripts

## Purpose

`gendec/scripts/` contains the checked-in shell entrypoints for the standard Phase 1 workflow:

- export the teacher dataset from ShapeNet into `gendec/data/ShapeNet`
- train the Phase 1 prior with validation enabled
- run held-out validation evaluation
- run held-out test evaluation

## Maintenance Contract

If a script is added, removed, or its invocation contract changes, this file must be updated in the same change.

## Files

### `common.sh`

- Shared script bootstrap.
- Resolves the script directory, resolves the repo root, and `cd`s to the repository before the concrete command runs.

### `export_shapenet_teacher.sh`

- Standard dataset-extraction script.
- Runs `python gendec/export_teacher.py --config gendec/configs/teacher_export.yaml`.
- Intended for the real SuperDec-to-`gendec/data/ShapeNet` export path.

### `train_phase1.sh`

- Standard Phase 1 training script.
- Runs `python gendec/train.py --config gendec/configs/train.yaml`.
- Uses the checked-in full-size training preset with validation, EMA, cosine scheduling, and training-time sampling diagnostics.
- If WandB logging is enabled, source `autodec/keys/keys.sh` or otherwise export `WANDB_API_KEY` before running the script.

### `eval_val.sh`

- Validation evaluation script.
- Runs `python gendec/eval/run.py --config gendec/configs/eval_val.yaml`.
- Intended for checkpoint selection and validation monitoring on the exported `val` split.

### `eval_test.sh`

- Test evaluation script.
- Runs `python gendec/eval/run.py --config gendec/configs/eval.yaml`.
- Intended for final held-out test evaluation on the exported `test` split.
