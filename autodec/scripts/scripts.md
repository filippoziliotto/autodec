# scripts

This folder contains shell entrypoints for running AutoDec training phases from
the repository root or from any other working directory.

Files:

```text
run_smoke.sh
run_phase1.sh
run_phase2.sh
scripts.md
```

All scripts:

- use `bash`;
- enable `set -euo pipefail`;
- resolve the repository root from the script location;
- `cd` to the repository root before launching Python;
- forward all extra command-line arguments with `"$@"`, so Hydra overrides can
  be appended at invocation time.

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

