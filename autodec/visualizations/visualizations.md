# `autodec/visualizations/`

This folder contains the AutoDec visualization utilities. The AutoDec trainer
can call them at the end of an evaluation epoch, then optionally upload the
same local files to WandB as 3D objects.

The default local output root is:

```text
data/viz/
```

For a run named `autodec_phase1`, epoch `3`, split `val`, and sample `0`, the
writer creates:

```text
data/viz/autodec_phase1/val/epoch_0003/sample_0000/
  input_gt.ply
  sq_mesh.obj
  reconstruction.ply
  metadata.json
```

## `__init__.py`

Exports the public visualization API:

```python
AutoDecEpochVisualizer
VisualizationRecord
build_wandb_log
log_wandb_visualizations
write_point_cloud_ply
export_sq_mesh
```

Import from the folder root when possible:

```python
from autodec.visualizations import AutoDecEpochVisualizer
```

## `epoch.py`

Defines the epoch-level API.

### `VisualizationRecord`

Dataclass returned by `AutoDecEpochVisualizer.write_epoch`.

Fields:

```text
epoch
split
sample_index
sample_dir
input_path
sq_mesh_path
reconstruction_path
metadata_path
```

The paths point to the local files written under `data/viz/`.

### `AutoDecEpochVisualizer`

Writes visualization files for one or more samples at the end of an epoch.

Constructor arguments:

```python
AutoDecEpochVisualizer(
    root_dir="data/viz",
    run_name="autodec",
    mesh_resolution=24,
    exist_threshold=0.5,
    max_points=4096,
    input_color=(180, 180, 180),
    reconstruction_color=(42, 157, 143),
)
```

Expected inputs:

```python
batch["points"]              # [B, N, 3] or [B, 3, N]
outdict["decoded_points"]    # [B, P*S_dec, 3]
outdict["scale"]             # [B, P, 3]
outdict["shape"]             # [B, P, 2]
outdict["rotate"]            # [B, P, 3, 3]
outdict["trans"]             # [B, P, 3]
outdict["exist"]             # [B, P, 1], or exist_logit instead
```

Usage:

```python
visualizer = AutoDecEpochVisualizer(
    root_dir="data/viz",
    run_name=cfg.run_name,
)
records = visualizer.write_epoch(
    batch=batch,
    outdict=outdict,
    epoch=epoch,
    split="val",
    num_samples=2,
)
```

This writes:

- `input_gt.ply`: the input or target point cloud.
- `sq_mesh.obj`: the active predicted superquadric scaffold as a mesh.
- `sq_mesh.mtl`: material colors for the scaffold, one material per active
  primitive color group.
- `reconstruction.ply`: the decoded AutoDec reconstruction point cloud.
- `metadata.json`: epoch, split, sample index, point counts, active primitive count.

### `build_wandb_log`

Builds a WandB payload but does not log it:

```python
payload = build_wandb_log(records)
```

Returned keys:

```text
visual/gt
visual/sq_mesh
visual/reconstruction
```

Each value is a list of `wandb.Object3D` objects. The function imports WandB
lazily, so the rest of the visualization package can be used without an active
WandB run. WandB does not accept PLY file paths for `Object3D`, so point-cloud
PLY files are converted to `[N, 6]` arrays before logging.

### `log_wandb_visualizations`

Convenience wrapper for later trainer integration:

```python
log_wandb_visualizations(wandb_run, records, step=epoch)
```

The trainer normally calls `build_wandb_log` through its injected
`wandb_visual_log_builder`; this wrapper is available for direct/manual use.

## `view_eval.py`

Launches a local browser-based viewer for test visualization folders written by
`AutoDecEpochVisualizer`:

```bash
python -m autodec.visualizations.view_eval data/eval/autodec_test_eval
```

The input path can be a run root, a split/epoch directory, or a single
`sample_0000` directory. Complete sample directories must contain:

```text
sq_mesh.obj
reconstruction.ply
input_gt.ply
```

The command starts three Viser panes plus one lightweight Flask wrapper page and
prints the wrapper URL. It does not open a browser automatically. The wrapper
page provides Back/Forward navigation and embeds independently interactive panes
for the superquadric mesh, decoded point reconstruction, and ground-truth point
cloud.

`viser` is imported lazily at runtime, so tests and non-viewer visualization
utilities can still import the module in environments where Viser is not
installed.

## `pointcloud.py`

Contains point-cloud formatting and local file export.

### `points_to_numpy`

Converts one sample to `[N, 3]` numpy format.

Accepted shapes:

```text
[B, N, 3]
[B, 3, N]
[N, 3]
[3, N]
```

If `max_points` is set and the input has more points than that, it keeps a
deterministic evenly spaced subset. This avoids random visual changes across
epochs.

### `write_point_cloud_ply`

Writes an ASCII PLY point cloud with RGB vertex colors:

```python
write_point_cloud_ply(
    path,
    points,
    color=(180, 180, 180),
    sample_index=0,
    max_points=4096,
)
```

The ground truth and reconstruction visualizations use this writer.

## `sq_mesh.py`

Contains the lightweight superquadric mesh exporter.

It does not call the SuperDec CUDA sampler or `PredictionHandler`. It directly
evaluates the signed-power superquadric surface on a regular `(eta, omega)`
grid, applies the predicted rotation and translation, and exports active
primitive meshes with deterministic colors. Mesh generation defensively clamps
shape exponents to `[0.1, 2.0]`, matching the sampler's valid exponent range.

### `build_sq_mesh`

Builds a combined `trimesh.Trimesh` from active primitives:

```python
mesh = build_sq_mesh(
    outdict,
    sample_index=0,
    resolution=24,
    exist_threshold=0.5,
)
```

The active mask uses `outdict["exist"]` when available, otherwise
`sigmoid(outdict["exist_logit"])`.

### `export_sq_mesh`

Writes the mesh to disk:

```python
export_sq_mesh(
    "data/viz/run/val/epoch_0001/sample_0000/sq_mesh.obj",
    outdict,
)
```

The current epoch visualizer writes mesh OBJ to avoid the `trimesh` GLB exporter
path, which is not compatible with NumPy 2 in some cluster environments, while
still using a mesh file type accepted by WandB `Object3D`. The OBJ references a
same-directory `.mtl` file with deterministic diffuse materials derived from
the primitive index, so active superquadrics render with different colors.
