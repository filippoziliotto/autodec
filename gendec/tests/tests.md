# tests

## Purpose

`gendec/tests/` contains the package-level unit and smoke tests for data contracts, geometry helpers, losses, checkpoints, documentation presence, and end-to-end toy execution.

## Maintenance Contract

If a new top-level test file is added here, or an existing test changes scope, this index must be updated in the same change.

## Files

### `test_checkpoints.py`

- `test_phase1_checkpoint_round_trip_restores_model_and_epoch`: verifies that saved checkpoints restore model weights and return stored epoch/loss metadata.

### `test_console_logger.py`

- `test_console_logger_formats_metric_map`: verifies metric dictionaries are rendered into stable console text.
- `_cfg(data_root, checkpoint_path, sample_dir)`: builds a small namespace config with tqdm disabled for console-output tests.
- `test_run_train_prints_epoch_metrics`: verifies the train entrypoint prints the end-of-epoch train/val/sample summary.

### `test_dataset.py`

- `test_toy_teacher_dataset_round_trips_schema_and_normalization`: checks that toy exports load through `ScaffoldTokenDataset` with the expected schema and normalized token fields.
- `test_saved_example_payload_matches_expected_keys`: checks that toy examples match the persisted example contract.
- `test_dataset_scans_category_and_model_directories`: verifies fallback directory scanning when loading exported examples.
- `test_dataset_uses_split_manifest_when_requested`: verifies split manifests are respected by the dataset loader.
- `test_test_split_export_keeps_existing_train_normalization_stats`: verifies test exports do not overwrite training normalization stats.
- `test_toy_export_can_materialize_train_val_test_splits_in_one_run`: verifies one toy export command can write all three split manifests and datasets.
- `test_toy_export_all_keyword_matches_train_val_test_splits`: verifies `split: all` expands to the canonical `train`/`val`/`test` export set.

### `test_docs.py`

- `test_source_directories_have_matching_folder_docs`: verifies the required folder-level markdown files exist for the maintained `gendec/` source directories.

### `test_flow_matching.py`

- `test_build_flow_batch_constructs_interpolation_and_target_velocity`: verifies the analytic interpolation path and target velocity.
- `test_flow_matching_loss_returns_flow_and_optional_existence_metrics`: verifies the main loss returns the expected metric keys and a tensor loss.

### `test_imports.py`

- `test_import_gendec_package_does_not_eagerly_import_eval_runtime`: verifies the package root remains importable without eagerly constructing the eval/runtime dependency surface.

### `test_model.py`

- `test_set_transformer_flow_model_matches_token_contract`: verifies the model input/output tensor shape contract.

### `test_ordering.py`

- `test_deterministic_sort_indices_follow_priority_rules`: verifies the deterministic ordering priority.
- `test_reorder_teacher_outputs_reorders_assignments_and_tokens_together`: verifies aligned reordering across teacher payload tensors.

### `test_preview_video.py`

- `_write_preview(path, points_scale)`: writes a minimal preview artifact for video tests.
- `test_collect_preview_epochs_selects_every_10`: verifies preview discovery keeps only the configured epoch stride.
- `test_build_preview_video_writes_video_under_run_name`: verifies the utility writes an MP4 to `gendec/videos/<run_name>/`.

### `test_rotation.py`

- `test_matrix_to_rot6d_uses_first_two_columns`: verifies the teacher-side 6D rotation conversion convention.
- `test_rot6d_to_matrix_returns_valid_rotation`: verifies that the inverse conversion produces orthonormal rotation matrices.

### `test_shapenet_index.py`

- `_write_example(root, category_id, model_id)`: local helper for writing minimal exported examples in filesystem tests.
- `test_iter_exported_examples_uses_split_manifest_when_present`: verifies exported split manifests control indexed examples.
- `test_scan_source_shapenet_models_falls_back_to_directory_listing`: verifies source ShapeNet scanning falls back to directory enumeration when manifests are missing.

### `test_smoke.py`

- `_cfg(data_root, checkpoint_path, sample_dir)`: builds a small namespace config for smoke execution.
- `test_toy_builder_train_and_sample_smoke`: verifies toy export, one-step training with validation/sample diagnostics, checkpointing, preview writing, and sampling work end to end.

### `test_wandb.py`

- `RecordingWandbRun`: lightweight fake WandB run used by the tests.
- `_cfg(data_root, checkpoint_path, sample_dir)`: builds a small namespace config with WandB enabled.
- `test_build_wandb_run_is_lazy_uses_project_and_env_key`: verifies lazy WandB initialization, project/name propagation, and environment-key forwarding.
- `test_build_wandb_run_returns_none_when_disabled`: verifies WandB is not imported when disabled.
- `test_run_train_logs_epoch_metrics_to_wandb`: verifies the train entrypoint logs train/val/sample metrics and finishes the WandB run.

### `test_scripts.py`

- `test_gendec_scripts_exist`: verifies the checked-in export/train/eval shell scripts are present in `gendec/scripts/`.

## Subfolder

- [`eval/eval.md`](./eval/eval.md): evaluation-specific tests, including the frozen AutoDec coarse-decode bridge and generated-SQ visualization artifacts.
