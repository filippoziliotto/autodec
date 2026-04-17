# `autodec/tests/visualizations/`

Tests for the AutoDec visualization utilities in `autodec/visualizations/`.

## `test_epoch_visualizer.py`

Verifies the local epoch visualization writer and the optional WandB payload
builder.

### `test_epoch_visualizer_writes_gt_sq_mesh_and_reconstruction`

Creates one tiny batch and one tiny AutoDec output dictionary.

Checks that `AutoDecEpochVisualizer.write_epoch` writes:

- `input_gt.ply`
- `sq_mesh.ply`
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

### `test_visualizations_folder_has_same_name_documentation`

Protects the convention that every AutoDec folder has a same-name Markdown
document describing its contents.
