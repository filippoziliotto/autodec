# data

## Purpose

`gendec/data/` owns the scaffold dataset contract:

- where exported teacher examples live
- how source ShapeNet models are scanned
- how teacher outputs are converted into scaffold tokens
- how dataset-level normalization is computed and loaded
- how toy datasets are generated for smoke tests

## Maintenance Contract

If any file in this folder changes its schema, indexing rules, or helper behavior, this document must be updated in the same change.

## Files

### `__init__.py`

- Re-export surface for the data layer.
- Exposes dataset loading, teacher-example IO, layout helpers, ordering helpers, pointcloud loading, ShapeNet scanning, and toy builders.

### `build_teacher_dataset.py`

- Real export path from raw ShapeNet inputs to serialized scaffold examples.
- `_lazy_load_superdec_model()`: local import wrapper for `SuperDec`.
- `_resolve_teacher_config_path(export_cfg)`: chooses the teacher config path, either explicit or sibling to the checkpoint.
- `_teacher_model(export_cfg)`: instantiates the frozen SuperDec model and loads checkpoint weights.
- `_resolved_export_splits(export_cfg)`: resolves `split` or `splits` into the ordered export split list.
- `_export_teacher_split(...)`: exports one requested source split into `gendec/data/ShapeNet/{cat}/{model_id}` and writes the matching manifest.
- `_output_stats(root, split, token_examples)`: writes normalization stats for train exports or first-time exports.
- `export_teacher_dataset(cfg)`: can now export `train`, `val`, and `test` in one run while keeping the source ShapeNet manifest boundaries intact.

### `dataset.py`

- Dataset loader used by training and evaluation.
- `ScaffoldTokenDataset`:
  - `__init__(root, split=None, categories=None)`: indexes exported examples, applies optional split/category filtering, and loads dataset normalization stats.
  - `__len__()`: returns number of indexed teacher examples.
  - `__getitem__(idx)`: loads one teacher example, returns raw tokens, normalized tokens, normalization stats, metadata, and auxiliary tensors.
- `load_normalization_stats`: re-export alias to the normalization loader.

### `examples.py`

- Teacher-example construction and serialization helpers.
- `_exist_logit(outdict)`: resolves `exist_logit` directly or derives it from `exist`.
- `build_teacher_example(outdict, points, model_id, category_id)`: converts teacher outputs into reordered scaffold tokens and the persisted example payload.
- `save_teacher_example(root, example)`: writes one example to `ShapeNet/{cat}/{model_id}/teacher_scaffold.pt`.
- `load_teacher_example(path)`: loads a serialized teacher example from disk.

### `layout.py`

- Filesystem conventions for exported datasets.
- `NORMALIZATION_FILENAME`, `SCAFFOLD_FILENAME`: canonical artifact names.
- `normalization_stats_path(root)`: returns `normalization.pt`.
- `model_dir(root, category_id, model_id)`: returns the per-model export directory.
- `scaffold_example_path(root, category_id, model_id)`: returns the `teacher_scaffold.pt` path.
- `split_manifest_path(root, category_id, split)`: returns the `{split}.lst` manifest path.
- `_read_manifest(path)`: parses manifest files into model-id lists.
- `iter_exported_examples(root, split=None, categories=None)`: yields indexed exported examples using manifests when available and directory scans otherwise.
- `write_split_manifest(root, split, model_index)`: writes per-category split manifests.

### `normalization.py`

- Channelwise normalization utilities for scaffold tokens.
- `_stats_for(tokens, stats)`: moves normalization stats onto the same device and dtype as the target tokens.
- `compute_normalization_stats(tokens)`: computes dataset mean and std over the last token dimension.
- `normalize_tokens(tokens, stats)`: applies `(tokens - mean) / std`.
- `unnormalize_tokens(tokens, stats)`: restores raw token values.
- `save_normalization_stats(path, stats)`: serializes normalization stats.
- `load_normalization_stats(path)`: loads normalization stats.

### `ordering.py`

- Offline deterministic primitive ordering.
- `compute_assignment_mass(assign_matrix)`: reduces the teacher assignment matrix into one mass score per primitive.
- `compute_primitive_volume(scale)`: computes `sx * sy * sz` for ordering purposes.
- `deterministic_sort_indices(exist, mass, volume, translation)`: sorts primitives by existence, mass, volume, then translation-x.
- `reorder_teacher_outputs(payload, order)`: applies the permutation consistently across primitive-aligned tensors and assignment columns.

### `pointclouds.py`

- Source ShapeNet pointcloud loading and per-instance normalization.
- `normalize_points(points)`: mean-centers and rescales a point cloud by twice its max absolute extent.
- `_pointcloud_file(model_dir)`: chooses the first available source pointcloud file among supported filenames.
- `load_source_pointcloud(model_dir, n_points=4096)`: loads, resamples or tiles to the requested count, normalizes, and returns points plus normalization metadata.

### `shapenet_index.py`

- Source dataset indexing for real teacher export.
- `_read_manifest(path)`: parses raw ShapeNet split files.
- `scan_source_shapenet_models(dataset_path, categories=None, split=None, require_manifest=False)`: yields category/model/source-directory records, using split manifests when present, and optionally fails loudly when a requested split manifest is missing.

### `splits.py`

- Shared split-resolution helpers for export and toy-data generation.
- `STANDARD_SHAPENET_SPLITS`: canonical ordered split tuple `("train", "val", "test")`.
- `resolve_split_names(split=None, splits=None, default="train")`: normalizes split requests, expands `all`, deduplicates, and returns the canonical ordered list.

### `toy_builder.py`

- Synthetic dataset generator for smoke tests and local end-to-end verification.
- `_rotation_6d()`: returns the identity rotation in 6D form.
- `_stable_seed(model_id, category_id)`: derives deterministic per-example random seeds.
- `build_toy_example(model_id, category_id="03001627", num_points=4096, n_primitives=PRIMITIVE_COUNT)`: creates one synthetic scaffold example matching the persisted schema.
- `write_toy_teacher_dataset(root, split="train", num_examples=8, num_points=4096)`: writes multiple toy examples, split manifests, and normalization stats.
- `write_toy_teacher_dataset_splits(root, split=None, splits=None, num_examples=8, num_points=4096)`: writes multiple toy splits in one run and returns a multi-split export summary.
