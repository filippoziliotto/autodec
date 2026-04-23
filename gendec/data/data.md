# data

## Purpose

`gendec/data/` owns the exported dataset contract:

- source ShapeNet scanning
- teacher output conversion
- offline deterministic ordering
- normalization-stat computation
- Phase 1 scaffold-token loading
- Phase 2 joint-token loading
- toy dataset generation for smoke tests

## Maintenance Contract

If any file in this folder changes its schema, indexing rules, or helper behavior, this document must be updated in the same change.

## Files

### `__init__.py`

- Re-export surface for the data layer.
- Exposes Phase 1 and Phase 2 dataset loading, teacher-example IO, layout helpers, ordering helpers, pointcloud loading, ShapeNet scanning, and toy builders.

### `build_teacher_dataset.py`

- Real export path from raw ShapeNet inputs to serialized teacher examples.
- `_lazy_load_superdec_model()`: lazy-import wrapper for `SuperDec`.
- `_lazy_load_autodec_model()`: lazy-import wrapper for `AutoDec` plus its checkpoint loader.
- `_resolve_teacher_config_path(export_cfg)`: chooses the teacher config path, either explicit or sibling to the checkpoint.
- `_superdec_teacher_model(export_cfg)`: instantiates and restores the frozen SuperDec teacher.
- `_autodec_teacher_model(export_cfg)`: instantiates and restores the frozen AutoDec teacher.
- `_teacher_model(export_cfg)`: dispatches between the SuperDec and AutoDec teacher loaders.
- `_stats_token_key(export_cfg)`: chooses whether normalization stats are computed over `tokens_e` or `tokens_ez`.
- `_run_teacher_model(model, export_cfg, points, device)`: runs the correct teacher forward path, using the AutoDec encoder directly for Phase 2 export.
- `_output_stats(root, split, token_examples)`: writes normalization stats for train exports or first-time exports.
- `_resolved_export_splits(export_cfg)`: resolves `split` or `splits` into the ordered export split list.
- `_export_teacher_split(...)`: exports one requested source split and writes the matching manifest.
- `export_teacher_dataset(cfg)`: exports Phase 1 scaffold-only or Phase 2 joint-token datasets while preserving the source split boundaries.

### `dataset.py`

- Dataset loaders used by training and evaluation.
- `ScaffoldTokenDataset`:
  - `__init__(root, split=None, categories=None)`: indexes exported Phase 1 examples, builds a stable category vocabulary, and loads normalization stats.
  - `category_ids`: sorted exported category ids visible to the dataset.
  - `category_to_index`: mapping from `category_id` to stable integer class index.
  - `num_classes`: number of visible classes.
  - `__len__()`: returns the number of indexed teacher examples.
  - `__getitem__(idx)`: returns raw scaffold tokens, normalized tokens, normalization stats, metadata, `category_index`, and auxiliary tensors.
- `JointTokenDataset`:
  - `__init__(root, split=None, categories=None)`: indexes exported Phase 2 examples, builds a stable category vocabulary, and loads joint-token normalization stats.
  - `category_ids`: sorted exported category ids visible to the dataset.
  - `category_to_index`: mapping from `category_id` to stable integer class index.
  - `num_classes`: number of visible classes.
  - `residual_dim`: inferred residual width from the normalization-stat width.
  - `__len__()`: returns the number of indexed joint teacher examples.
  - `__getitem__(idx)`: returns normalized `tokens_ez`, raw `tokens_ez_raw`, split `tokens_e`, split `tokens_z`, normalization stats, metadata, `category_index`, and auxiliary tensors.
- `load_normalization_stats`: re-export alias to the normalization loader.

### `examples.py`

- Teacher-example construction and serialization helpers.
- `_exist_logit(outdict)`: resolves `exist_logit` directly or derives it from `exist`.
- `build_teacher_example(outdict, points, model_id, category_id)`: converts teacher outputs into reordered scaffold tokens and, when the teacher emits `residual`, also builds reordered `tokens_z` and `tokens_ez`.
- `save_teacher_example(root, example)`: writes one example to `ShapeNet.../teacher_scaffold.pt`.
- `load_teacher_example(path)`: loads a serialized teacher example from disk.

### `layout.py`

- Filesystem conventions for exported datasets.
- `NORMALIZATION_FILENAME`, `SCAFFOLD_FILENAME`: canonical artifact names.
- `normalization_stats_path(root)`: returns `normalization.pt`.
- `model_dir(root, category_id, model_id)`: returns the per-model export directory.
- `scaffold_example_path(root, category_id, model_id)`: returns the `teacher_scaffold.pt` path.
- `split_manifest_path(root, category_id, split)`: returns the `{split}.lst` manifest path.
- `_read_manifest(path)`: parses manifest files into model-id lists.
- `available_categories(root, categories=None)`: resolves the visible exported category ids from disk or an explicit filter.
- `build_category_vocab(root, categories=None)`: returns the stable sorted category list plus `category_id -> index` mapping.
- `iter_exported_examples(root, split=None, categories=None)`: yields indexed exported examples using manifests when available and directory scans otherwise.
- `write_split_manifest(root, split, model_index)`: writes per-category split manifests.

### `normalization.py`

- Channelwise normalization utilities for token tensors.
- `_stats_for(tokens, stats)`: moves normalization stats onto the same device and dtype as the target tokens.
- `compute_normalization_stats(tokens)`: computes dataset mean and std over the last token dimension.
- `normalize_tokens(tokens, stats)`: applies `(tokens - mean) / std`.
- `unnormalize_tokens(tokens, stats)`: restores raw token values.
- `save_normalization_stats(path, stats)`: serializes normalization stats.
- `load_normalization_stats(path)`: loads normalization stats.

### `ordering.py`

- Offline deterministic primitive ordering.
- `compute_assignment_mass(assign_matrix)`: reduces the teacher assignment matrix into one mass score per primitive.
- `compute_primitive_volume(scale)`: computes `sx * sy * sz` for ordering.
- `deterministic_sort_indices(exist, mass, volume, translation)`: sorts primitives by existence, mass, volume, then translation-x.
- `reorder_teacher_outputs(payload, order)`: applies the permutation consistently across primitive-aligned tensors and assignment columns.

### `pointclouds.py`

- Source ShapeNet pointcloud loading and per-instance normalization.

### `shapenet_index.py`

- Source dataset indexing for real teacher export.

### `splits.py`

- Shared split-resolution helpers for export and toy-data generation.

### `toy_builder.py`

- Synthetic dataset generator for smoke tests and local verification.
- `_rotation_6d()`: returns the identity rotation in 6D form.
- `_stable_seed(model_id, category_id)`: derives deterministic per-example random seeds.
- `build_toy_example(...)`: creates one synthetic Phase 1 scaffold example.
- `build_toy_phase2_example(...)`: creates one synthetic Phase 2 joint-token example with `tokens_e`, `tokens_z`, and `tokens_ez`.
- `write_toy_teacher_dataset(...)`: writes multiple toy Phase 1 examples plus manifests and normalization stats.
- `write_toy_phase2_dataset(...)`: writes multiple toy Phase 2 examples plus manifests and joint normalization stats.
- `write_toy_teacher_dataset_splits(...)`: writes multiple toy Phase 1 splits in one run.
- `write_toy_phase2_dataset_splits(...)`: writes multiple toy Phase 2 splits in one run.
