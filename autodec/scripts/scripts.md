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
scripts.md
```

All scripts:

- use `bash`;
- enable `set -euo pipefail`;
- resolve the repository root from the script location;
- `cd` to the repository root before launching Python;
- call `ensure_fast_sampler` before training, so the SuperDec Cython/C++
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
