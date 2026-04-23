from pathlib import Path
from types import SimpleNamespace


def _cfg(data_root, checkpoint_path, sample_dir):
    return SimpleNamespace(
        seed=0,
        run_name="gendec_smoke",
        dataset=SimpleNamespace(root=str(data_root), split="train", val_split="val"),
        model=SimpleNamespace(token_dim=15, hidden_dim=32, n_blocks=2, n_heads=4, dropout=0.0),
        loss=SimpleNamespace(lambda_flow=1.0, lambda_exist=0.05, exist_channel=-1),
        optimizer=SimpleNamespace(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-8,
        ),
        scheduler=SimpleNamespace(name="cosine", warmup_steps=0, min_lr=1e-5),
        training=SimpleNamespace(
            batch_size=2,
            num_workers=0,
            num_epochs=1,
            checkpoint_path=str(checkpoint_path),
            best_checkpoint_path=str(checkpoint_path.with_name("best.pt")),
            metrics_path=str(checkpoint_path.with_suffix(".jsonl")),
            grad_clip_norm=1.0,
            ema_decay=0.999,
            amp=False,
            preview_every=1,
            preview_dir=str(sample_dir / "train_previews"),
        ),
        sampling=SimpleNamespace(
            preview_steps=4,
            eval_steps=4,
            exist_threshold=0.5,
            preview_num_samples=2,
            num_samples=2,
            output_dir=str(sample_dir),
        ),
    )


def test_toy_builder_train_and_sample_smoke(tmp_path):
    from gendec.data.toy_builder import write_toy_teacher_dataset
    from gendec.sample import run_sample
    from gendec.train import run_train

    root = tmp_path / "ShapeNet"
    checkpoint_path = tmp_path / "phase1.pt"
    sample_dir = tmp_path / "samples"
    write_toy_teacher_dataset(root=root, split="train", num_examples=4, num_points=64)
    write_toy_teacher_dataset(root=root, split="val", num_examples=2, num_points=64)
    write_toy_teacher_dataset(root=root, split="test", num_examples=2, num_points=64)

    cfg = _cfg(root, checkpoint_path, sample_dir)
    train_result = run_train(cfg)
    sample_result = run_sample(cfg)

    assert Path(train_result["checkpoint_path"]).is_file()
    assert Path(train_result["best_checkpoint_path"]).is_file()
    assert "val_metrics" in train_result
    assert "sample_metrics" in train_result
    assert "active_primitive_count_mean" in train_result["sample_metrics"]
    assert "exist_entropy" in train_result["val_metrics"]
    assert "field_mse_scale" in train_result["val_metrics"]
    assert "positive_scale_fraction" in train_result["sample_metrics"]
    assert Path(train_result["preview_path"]).is_file()
    assert sample_result["tokens"].shape == (2, 16, 15)
    assert sample_result["active_mask"].shape == (2, 16)
    assert sample_result["preview_points"].ndim == 3
