from pathlib import Path
from types import SimpleNamespace


def test_console_logger_formats_metric_map():
    from gendec.utils.logger import TrainingConsoleLogger

    logger = TrainingConsoleLogger(disable_tqdm=True)

    text = logger.format_metrics({"all": 1.234567, "flow_loss": 0.5})

    assert "all=1.2346" in text
    assert "flow_loss=0.5000" in text


def _cfg(data_root, checkpoint_path, sample_dir):
    return SimpleNamespace(
        seed=0,
        run_name="gendec_smoke_console",
        use_wandb=False,
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
            disable_tqdm=True,
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


def test_run_train_prints_epoch_metrics(tmp_path, capsys):
    from gendec.data.toy_builder import write_toy_teacher_dataset
    from gendec.train import run_train

    root = tmp_path / "ShapeNet"
    checkpoint_path = tmp_path / "phase1.pt"
    sample_dir = tmp_path / "samples"
    write_toy_teacher_dataset(root=root, split="train", num_examples=4, num_points=64)
    write_toy_teacher_dataset(root=root, split="val", num_examples=2, num_points=64)

    run_train(_cfg(root, checkpoint_path, sample_dir))
    captured = capsys.readouterr()

    assert "Epoch 1/1" in captured.out
    assert "train:" in captured.out
    assert "val:" in captured.out
    assert "samples:" in captured.out


def test_run_train_phase2_prints_epoch_metrics(tmp_path, capsys):
    from gendec.data.toy_builder import write_toy_phase2_dataset
    from gendec.train_phase2 import run_train_phase2

    root = tmp_path / "ShapeNetPhase2"
    checkpoint_path = tmp_path / "phase2.pt"
    sample_dir = tmp_path / "samples_phase2"
    write_toy_phase2_dataset(root=root, split="train", num_examples=4, num_points=64, residual_dim=4)
    write_toy_phase2_dataset(root=root, split="val", num_examples=2, num_points=64, residual_dim=4)

    cfg = SimpleNamespace(
        seed=0,
        run_name="gendec_phase2_console",
        use_wandb=False,
        dataset=SimpleNamespace(root=str(root), split="train", val_split="val"),
        model=SimpleNamespace(explicit_dim=15, residual_dim=4, hidden_dim=32, n_blocks=2, n_heads=4, dropout=0.0),
        loss=SimpleNamespace(explicit_dim=15, lambda_e=1.0, lambda_z=1.0, lambda_exist=0.05, exist_channel=-1),
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
            disable_tqdm=True,
        ),
        sampling=SimpleNamespace(
            preview_steps=4,
            eval_steps=4,
            exist_threshold=0.5,
            preview_num_samples=2,
            num_samples=2,
            output_dir=str(sample_dir),
        ),
        checkpoints=SimpleNamespace(resume_from=None),
    )

    run_train_phase2(cfg)
    captured = capsys.readouterr()

    assert "Epoch 1/1" in captured.out
    assert "train:" in captured.out
    assert "val:" in captured.out
    assert "samples:" in captured.out
