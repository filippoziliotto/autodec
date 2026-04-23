# configs

## Purpose

`gendec/configs/` contains YAML runtime presets for export, training, sampling utilities, smoke runs, and held-out evaluation across both Phase 1 and Phase 2.

## Maintenance Contract

When any config file changes, this file must be updated to reflect the new purpose or key layout.

## Files

### `eval.yaml`

- Main evaluation preset for a trained full-size Phase 1 model.
- Evaluates `gendec/data/ShapeNet` on `split: test`.
- Includes optional zero-residual AutoDec decode for coarse plausibility metrics.
- Includes a `conditioning` block for optional class conditioning and `eval.generated_per_class` for class-conditioned test generation.

### `eval_val.yaml`

- Validation evaluation preset for Phase 1.
- Inherits `eval.yaml`, switches to `dataset.split: val`, and disables test-only visualization export.

### `eval_test.yaml`

- Lightweight smoke evaluation preset for small Phase 1 checkpoints.
- Uses the small debug model and disabled AutoDec decoding for fast local verification.

### `eval_phase2.yaml`

- Main evaluation preset for a trained full-size Phase 2 model.
- Evaluates `gendec/data/ShapeNetPhase2` on `split: test`.
- Uses the joint-token model config (`explicit_dim=15`, `residual_dim=64`) and split explicit/residual loss settings.
- Enables the frozen AutoDec decode branch by default, using:
  - `checkpoints/autodec_phase2_ddp3_200ep_bs32_from_p1_cons01_ep50_cons01/config.yaml`
  - `checkpoints/autodec_phase2_ddp3_200ep_bs32_from_p1_cons01_ep50_cons01/epoch_200.pt`
- Includes a `conditioning` block for optional class conditioning and `eval.generated_per_class` for class-conditioned test generation.

### `eval_phase2_val.yaml`

- Validation evaluation preset for Phase 2.
- Inherits `eval_phase2.yaml`, switches to `dataset.split: val`, disables visualization export, and disables the frozen AutoDec decode branch for lighter validation runs.

### `eval_phase2_test.yaml`

- Lightweight smoke evaluation preset for small Phase 2 checkpoints.
- Uses the small debug model and keeps the Phase 2 AutoDec decode branch disabled by default for fast local verification.

### `sample.yaml`

- Sampling-only preset for unconditional Phase 1 generation from a trained checkpoint.
- Uses `dataset.root` only to locate normalization stats and checkpoint-compatible token dimensions.
- Includes a `conditioning` block so conditioned checkpoints can cycle through class indices at sampling time.

### `preview_video.yaml`

- Utility preset for turning saved training preview checkpoints into per-sample videos.
- `preview_video.preview_dir`: folder containing `epoch_XXXX_preview.pt` artifacts.
- `preview_video.run_name`: output subfolder name under `gendec/videos/`.
- `preview_video.output_root`: video root directory.
- `preview_video.every_n_epochs`: frame stride over saved preview epochs.
- `preview_video.fps`: output video frame rate.
- `preview_video.sample_index`: starting sampled object index inside each preview artifact to render.
- `preview_video.num_videos`: how many per-sample MP4 files to create, named `video_000000.mp4`, `video_000001.mp4`, and so on.

### `smoke.yaml`

- End-to-end smoke preset used for toy Phase 1 export, training, and sampling verification.
- Keeps class conditioning disabled by default.

### `teacher_export.yaml`

- Real Phase 1 teacher-export preset.
- Uses `export.mode: real` and the frozen SuperDec teacher.
- Writes scaffold-only examples under `gendec/data/ShapeNet`.

### `phase2_export.yaml`

- Real Phase 2 teacher-export preset.
- Uses `export.mode: real` plus `export.teacher_kind: autodec`.
- Writes joint-token examples under `gendec/data/ShapeNetPhase2`.
- Defaults the frozen AutoDec teacher checkpoint to:
  - `checkpoints/autodec_phase2_ddp3_200ep_bs32_from_p1_cons01_ep50_cons01/epoch_200.pt`

### `toy_teacher_export.yaml`

- Toy export preset for local Phase 1 verification.
- Uses `export.mode: toy`.

### `toy_phase2_export.yaml`

- Toy export preset for local Phase 2 verification.
- Uses `export.mode: phase2_toy`.
- Writes synthetic `tokens_e`, `tokens_z`, and `tokens_ez` under `gendec/data/ShapeNetPhase2`.

### `train.yaml`

- Main Phase 1 training preset.
- Enables WandB by default.
- Uses AdamW, cosine warmup, EMA, preview sampling, and last/best checkpoint writing.
- Includes a `conditioning` block that can enable learned class embeddings when training on multi-class exported roots.

### `train_phase2.yaml`

- Main Phase 2 training preset.
- Enables WandB by default.
- Trains on `gendec/data/ShapeNetPhase2` with the joint-token model and split explicit/residual loss.
- Writes:
  - `gendec/data/checkpoints/phase2_last.pt`
  - `gendec/data/checkpoints/phase2_best.pt`
  - `gendec/data/checkpoints/phase2_metrics.jsonl`
- Uses `gendec/data/previews_phase2` for training-time preview artifacts.
- Includes a `conditioning` block that can enable learned class embeddings when training on multi-class exported roots.
