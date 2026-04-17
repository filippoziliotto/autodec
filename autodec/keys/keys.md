# `autodec/keys/`

This folder currently contains local shell helpers for environment variables.

## `keys.sh`

Shell script intended to export local secrets such as the WandB API key before
launching training.

This file must be treated as machine-local secret material:

- do not commit it;
- do not paste its contents into logs or documentation;
- prefer environment variables, a local `.env`, or the standard `wandb login`
  flow on the training machine.

The repository `.gitignore` ignores `autodec/keys/keys.sh`.

AutoDec training does not import files from this folder. Source the file in the
shell before launching training if you want it to provide `WANDB_API_KEY`:

```bash
source autodec/keys/keys.sh
```

WandB integration reads the active process environment and uses
`wandb.init(project="autodec", name=run_name)` when `use_wandb: true`.
