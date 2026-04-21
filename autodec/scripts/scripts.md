# scripts

This folder contains shell entrypoints for running AutoDec training phases from
the repository root or from any other working directory.

Files:

```text
common.sh
run_smoke.sh
run_phase1.sh
run_phase2.sh
run_all.sh
run_eval_test.sh
run_multigpu_pipeline.sh
scripts.md
```

All scripts:

- use `bash`;
- enable `set -euo pipefail`;
- resolve the repository root from the script location;
- `cd` to the repository root before launching Python;
- call `ensure_fast_sampler` before training/evaluation, so the SuperDec Cython/C++
  sampler is rebuilt when a bind-mounted repo hides the compiled extension;
- forward all extra command-line arguments with `"$@"`, so Hydra overrides can
  be appended at invocation time.

## `common.sh`

Shared shell helpers for the AutoDec launch scripts.

### `ensure_fast_sampler`

Checks whether Python can resolve:

```text
superdec.fast_sampler._sampler
```

If the extension is available, the function returns immediately. If not, it
runs:

```bash
python setup_sampler.py build_ext --inplace
```

This extension is required by `superdec.loss.sampler.EqualDistanceSamplerSQ`.
Docker images may already contain the compiled `.so`, but mounting the repo from
the host can hide that build artifact. Rebuilding on demand keeps the phase
scripts usable in normal local environments and bind-mounted container runs.

## `run_smoke.sh`

Command:

```bash
python -m autodec.training.train --config-name smoke "$@"
```

Purpose:

Run the smallest AutoDec training config:

```text
autodec/configs/smoke.yaml
```

This is intended for checking that the AutoDec training stack can start before
launching a real phase run.

Example:

```bash
bash autodec/scripts/run_smoke.sh trainer.num_epochs=1 trainer.batch_size=1
```

## `run_phase1.sh`

Command:

```bash
python -m autodec.training.train --config-name train_phase1 "$@"
```

Purpose:

Run phase 1 decoder warmup:

```text
autodec/configs/train_phase1.yaml
```

Phase 1 freezes the pretrained SuperDec backbone and trains:

```text
encoder.residual_projector
decoder
```

Expected use:

```bash
bash autodec/scripts/run_phase1.sh
```

Useful override example:

```bash
bash autodec/scripts/run_phase1.sh trainer.num_epochs=2 trainer.batch_size=1
```

## `run_phase2.sh`

Command:

```bash
python -m autodec.training.train --config-name train_phase2 "$@"
```

Purpose:

Run phase 2 joint fine-tuning:

```text
autodec/configs/train_phase2.yaml
```

Phase 2 resumes a full AutoDec checkpoint, unfreezes the encoder, and trains
with reconstruction plus SQ/parsimony/existence regularizers.

Expected use:

```bash
bash autodec/scripts/run_phase2.sh
```

Useful override example:

```bash
bash autodec/scripts/run_phase2.sh checkpoints.resume_from=checkpoints/autodec_phase1/epoch_10.pt
```

## `run_all.sh`

Runs the two standard training phases sequentially:

```bash
bash autodec/scripts/run_phase1.sh "$@"
bash autodec/scripts/run_phase2.sh "$@"
```

Because the script uses `set -euo pipefail`, phase 2 starts only if phase 1
exits successfully. Extra command-line arguments are forwarded to both phases,
which is useful for shared overrides such as:

```bash
bash autodec/scripts/run_all.sh use_wandb=true wandb.project=autodec
```

## `run_eval_test.sh`

Command:

```bash
python -m autodec.eval.run --config-name eval_test "$@"
```

Purpose:

Run standalone AutoDec evaluation on the ShapeNet `test` split:

```text
autodec/configs/eval_test.yaml
```

The script loads a full AutoDec checkpoint, computes test metrics, writes
`metrics.json` and `per_sample_metrics.jsonl`, and exports the configured 3D
visualizations under `data/eval/<run_name>/`.

Expected use:

```bash
bash autodec/scripts/run_eval_test.sh checkpoints.resume_from=checkpoints/autodec_phase2/epoch_200.pt
```

Useful all-category WandB example:

```bash
bash autodec/scripts/run_eval_test.sh \
  checkpoints.resume_from=checkpoints/autodec_phase2/epoch_200.pt \
  use_wandb=true \
  wandb.project=autodec
```

For the SuperDec out-of-category held-out classes, use:

```bash
python -m autodec.eval.run \
  --config-name eval_test_out_category \
  checkpoints.resume_from=checkpoints/autodec_phase2/epoch_200.pt
```

## `run_multigpu_pipeline.sh`

Runs the standard AutoDec pipeline with distributed training:

```bash
torchrun --nproc_per_node="${NUM_GPUS}" -m autodec.training.train --config-name train_phase1 ...
torchrun --nproc_per_node="${NUM_GPUS}" -m autodec.training.train --config-name train_phase2 ...
python -m autodec.eval.run --config-name eval_test ...
```

Defaults:

```text
PHASE1_EPOCHS=100
PHASE2_EPOCHS=200
BATCH_SIZE_PER_GPU=8
NUM_GPUS=<torch.cuda.device_count()>
CHECKPOINT_ROOT=checkpoints
PHASE1_RUN_NAME=autodec_phase1_ddp${NUM_GPUS}_100ep_bs8
PHASE2_RUN_NAME=autodec_phase2_ddp${NUM_GPUS}_200ep_bs8
EVAL_RUN_NAME=autodec_test_eval_ddp${NUM_GPUS}_phase2_200ep_bs8
```

Phase 2 automatically resumes:

```text
${CHECKPOINT_ROOT}/${PHASE1_RUN_NAME}/epoch_${PHASE1_EPOCHS}.pt
```

Evaluation automatically loads:

```text
${CHECKPOINT_ROOT}/${PHASE2_RUN_NAME}/epoch_${PHASE2_EPOCHS}.pt
```

Training uses `torchrun` for DDP. Evaluation runs once in a single process
because the current evaluator is not distributed; launching it with `torchrun`
would duplicate the same test pass and write the same output files from
multiple processes.

Example:

```bash
bash autodec/scripts/run_multigpu_pipeline.sh
```

Useful environment overrides:

```bash
NUM_GPUS=4 BATCH_SIZE_PER_GPU=8 bash autodec/scripts/run_multigpu_pipeline.sh
```

Extra Hydra overrides are forwarded to phase 1, phase 2, and evaluation:

```bash
NUM_GPUS=4 bash autodec/scripts/run_multigpu_pipeline.sh use_wandb=true wandb.project=autodec
```
