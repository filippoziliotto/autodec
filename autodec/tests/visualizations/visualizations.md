# `autodec/tests/visualizations/`

Tests for the AutoDec visualization utilities in `autodec/visualizations/`.

## `test_epoch_visualizer.py`

Verifies the local epoch visualization writer and the optional WandB payload
builder.

## `test_view_eval.py`

Verifies the pure, non-server pieces of the local test visualization browser:

- recursive sample discovery from run, epoch, or direct sample paths
- complete-sample validation for `sq_mesh.obj`, `reconstruction.ply`, and
  `input_gt.ply`
- optional `metadata.json` loading
- CLI argument parsing without importing Viser
- wrapper HTML containing three panes and Back/Forward endpoints

### `test_epoch_visualizer_writes_gt_sq_mesh_and_reconstruction`

Creates one tiny batch and one tiny AutoDec output dictionary.

Checks that `AutoDecEpochVisualizer.write_epoch` writes:

- `input_gt.ply`
- `sq_mesh.obj`
- optional `sq_mesh_lm.obj`
- `reconstruction.ply`
- `metadata.json`

Also verifies metadata fields for epoch, split, sample index, point counts, and
active primitive count.

### `test_build_wandb_log_returns_expected_visual_keys`

Uses a fake `Object3D` factory, so no WandB process is required.

Checks that `build_wandb_log` returns:

```text
visual/gt
visual/sq_mesh
visual/reconstruction
```

and that the object factory receives the local visualization file paths.

### `test_build_wandb_log_omits_saved_lm_sq_mesh`

Checks that optional `sq_mesh_lm.obj` files are still saved locally but are not
included in the WandB payload.

### `test_sq_mesh_vertices_clamp_out_of_range_shape_exponents`

Passes out-of-range shape exponents directly to the SQ mesh helper and verifies
that exported vertices use the defensive `[0.1, 2.0]` exponent clamp.

### `test_visualizations_folder_has_same_name_documentation`

Protects the convention that every AutoDec folder has a same-name Markdown
document describing its contents.
