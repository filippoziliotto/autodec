import os
from pathlib import Path
from types import SimpleNamespace


class RecordingWandbRun:
    def __init__(self):
        self.logs = []
        self.finished = False

    def log(self, payload, step=None):
        self.logs.append((payload, step))

    def finish(self):
        self.finished = True


def _cfg(data_root, checkpoint_path, sample_dir):
    return SimpleNamespace(
        seed=0,
        run_name="gendec_wandb_test",
        use_wandb=True,
        wandb=SimpleNamespace(project="gendec-test", api_key_env="WANDB_SECRET"),
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
            preview_num_samples=2,
            preview_steps=4,
            eval_steps=4,
            exist_threshold=0.5,
            output_dir=str(sample_dir),
        ),
    )


def test_build_wandb_run_is_lazy_uses_project_and_env_key(monkeypatch):
    from gendec.training.builders import build_wandb_run

    calls = {}

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            calls["kwargs"] = kwargs
            return "RUN"

    monkeypatch.setattr("gendec.training.builders._import_wandb", lambda: FakeWandb)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_SECRET", "secret")

    cfg = SimpleNamespace(
        use_wandb=True,
        run_name="gendec_phase1",
        wandb=SimpleNamespace(project="gendec", api_key_env="WANDB_SECRET"),
    )

    run = build_wandb_run(cfg)

    assert run == "RUN"
    assert calls["kwargs"] == {"project": "gendec", "name": "gendec_phase1"}
    assert os.environ["WANDB_API_KEY"] == "secret"


def test_build_wandb_run_returns_none_when_disabled(monkeypatch):
    from gendec.training.builders import build_wandb_run

    monkeypatch.setattr(
        "gendec.training.builders._import_wandb",
        lambda: (_ for _ in ()).throw(AssertionError("wandb should not import")),
    )

    assert build_wandb_run(SimpleNamespace(use_wandb=False)) is None


def test_run_train_logs_epoch_metrics_to_wandb(tmp_path, monkeypatch):
    from gendec.data.toy_builder import write_toy_teacher_dataset
    from gendec.train import run_train

    root = tmp_path / "ShapeNet"
    checkpoint_path = tmp_path / "phase1.pt"
    sample_dir = tmp_path / "samples"
    write_toy_teacher_dataset(root=root, split="train", num_examples=4, num_points=64)
    write_toy_teacher_dataset(root=root, split="val", num_examples=2, num_points=64)

    wandb_run = RecordingWandbRun()
    monkeypatch.setattr("gendec.train.build_wandb_run", lambda cfg: wandb_run)

    cfg = _cfg(root, checkpoint_path, sample_dir)
    result = run_train(cfg)

    assert Path(result["checkpoint_path"]).is_file()
    assert wandb_run.finished is True
    assert len(wandb_run.logs) == 1
    payload, step = wandb_run.logs[0]
    assert step == 0
    assert "train/all" in payload
    assert "val/all" in payload
    assert "samples/active_primitive_count_mean" in payload
