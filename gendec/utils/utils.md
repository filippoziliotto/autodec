# utils

## Purpose

`gendec/utils/` contains small reusable helpers that do not belong to a single training, data, or model module. It currently owns console logging, generated-sample visualization helpers, and the preview-video utility.

## Maintenance Contract

If a file is added here or the logging/runtime helper behavior changes, this document must be updated in the same change.

## Files

### `__init__.py`

- Re-export surface for the utils package.
- Exposes `TrainingConsoleLogger`, `GeneratedSQVisualizer`, preview-video helpers, and point-pruning helpers for active-primitive inference outputs.

### `inference.py`

- Active-primitive pruning helpers for generated point clouds.
- `_exist_probability(outdict)`: resolves per-primitive existence probability from `exist` when present, otherwise from `sigmoid(exist_logit)`.
- `_resample_points(points, target_count)`: deterministically downsamples or repeats a variable-length point cloud to a fixed size.
- `prune_points_by_active_primitives(outdict, points_key, exist_threshold=0.5, target_count=None)`: keeps only points whose `part_ids` belong to primitives above the existence threshold.
- `prune_decoded_points(outdict, exist_threshold=0.5, target_count=None)`: convenience wrapper for pruning `decoded_points`.

### `logger.py`

- Console logging helper for human-readable training progress.
- `TrainingConsoleLogger`:
  - `__init__(disable_tqdm=False)`: stores whether tqdm progress bars should be disabled.
  - `format_metrics(metrics)`: formats a metric dict into sorted `key=value` text for console summaries.
  - `progress_bar(iterable, *, desc, leave=False)`: wraps an iterable in `tqdm.auto.tqdm` when available, or returns the iterable unchanged.
  - `update_progress_postfix(progress, metrics)`: pushes the latest batch metrics into a tqdm postfix when supported.
  - `print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics, sample_metrics)`: prints the end-of-epoch train/val/sample summary lines.

### `visualization.py`

- Generated-SQ visualization helpers used by test evaluation.
- `GeneratedSQVisualizationRecord`: dataclass describing one saved generated-sample visualization folder.
- `GeneratedSQVisualizer`:
  - `__init__(root_dir="data/viz", run_name="gendec_eval", mesh_resolution=24, exist_threshold=0.5, max_preview_points=4096)`: stores visualization output settings.
  - `_sample_dir(split, sample_index)`: returns the per-generated-sample directory under `data/viz/<run_name>/<split>/`.
  - `_write_metadata(path, split, sample_index, preview_points, active_primitives, decoded_points=None)`: writes summary metadata for one generated SQ and optionally records the decoded-point count when a frozen AutoDec reconstruction is available.
  - `_export_sq_mesh(path, processed, sample_index)`: writes the generated active SQ scaffold mesh as OBJ plus MTL.
  - `write_generated(processed, split="test", num_samples=10, decoded_points=None)`: writes `sq_mesh.obj`, `preview_points.ply`, `metadata.json`, and, when `decoded_points` is provided, `decoded_points.ply` for generated samples. `decoded_points` may be either a dense tensor or a per-sample list of already-pruned point clouds.
- `write_point_cloud_ply(path, points, color=(210, 210, 210), max_points=None)`: writes a preview point cloud PLY file.

### `preview_video.py`

- Utility for turning saved training preview artifacts into a rendered video.
- `_epoch_from_path(path)`: extracts the epoch number from `epoch_XXXX_preview.pt`.
- `collect_preview_epochs(preview_dir, every_n_epochs=10)`: finds preview files and keeps only epochs that match the configured stride.
- `_points_from_preview(path, sample_index=0)`: loads one preview artifact and extracts one generated preview point cloud.
- `_frame_from_points(points, epoch, image_size=(640, 640))`: renders one 3D scatter frame with matplotlib.
- `_build_single_preview_video(preview_dir, video_path, selected, fps=4, sample_index=0)`: writes one MP4 for one chosen sample index across the selected preview epochs.
- `build_preview_video(preview_dir, run_name, output_root="gendec/videos", every_n_epochs=10, fps=4, sample_index=0, num_videos=10)`: writes `num_videos` MP4 files under `gendec/videos/<run_name>/`, named `video_000000.mp4`, `video_000001.mp4`, and so on. Each video tracks a different sampled object index across the selected preview epochs.
- `_run_from_cfg(cfg)`: CLI entrypoint for running the preview-video utility from YAML config.
