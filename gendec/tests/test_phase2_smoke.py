from pathlib import Path


def test_phase2_toy_export_train_and_eval_smoke(tmp_path):
    from gendec.config import to_namespace
    from gendec.export_teacher import run_export
    from gendec.eval.run_phase2 import run_eval_phase2
    from gendec.train_phase2 import run_train_phase2

    export_root = tmp_path / "phase2_data"
    export_cfg = to_namespace(
        {
            "export": {
                "mode": "phase2_toy",
                "output_root": str(export_root),
                "splits": ["train", "val", "test"],
                "num_examples": 4,
                "num_points": 128,
                "residual_dim": 64,
            }
        }
    )
    export_result = run_export(export_cfg)
    assert export_result["root"] == export_root
    assert (export_root / "normalization.pt").is_file()

    checkpoint_dir = tmp_path / "checkpoints"
    train_cfg = to_namespace(
        {
            "seed": 0,
            "run_name": "phase2_smoke",
            "use_wandb": False,
            "dataset": {
                "root": str(export_root),
                "split": "train",
                "val_split": "val",
            },
            "model": {
                "explicit_dim": 15,
                "residual_dim": 64,
                "hidden_dim": 32,
                "n_blocks": 2,
                "n_heads": 4,
                "dropout": 0.0,
            },
            "loss": {
                "explicit_dim": 15,
                "lambda_e": 1.0,
                "lambda_z": 1.0,
                "lambda_exist": 0.05,
            },
            "optimizer": {
                "name": "AdamW",
                "lr": 1e-4,
            },
            "training": {
                "batch_size": 2,
                "num_workers": 0,
                "num_epochs": 1,
                "disable_tqdm": True,
                "amp": False,
                "checkpoint_path": str(checkpoint_dir / "phase2_last.pt"),
                "best_checkpoint_path": str(checkpoint_dir / "phase2_best.pt"),
                "metrics_path": str(checkpoint_dir / "phase2_metrics.jsonl"),
                "preview_every": 1,
                "preview_dir": str(tmp_path / "previews_phase2"),
            },
            "sampling": {
                "preview_num_samples": 2,
                "preview_steps": 4,
                "eval_steps": 4,
                "exist_threshold": 0.5,
            },
            "checkpoints": {"resume_from": None},
        }
    )
    train_result = run_train_phase2(train_cfg)
    assert Path(train_result["checkpoint_path"]).is_file()
    assert Path(train_result["best_checkpoint_path"]).is_file()
    assert Path(train_result["preview_path"]).is_file()

    eval_cfg = to_namespace(
        {
            "seed": 0,
            "run_name": "phase2_smoke_eval",
            "dataset": {
                "root": str(export_root),
                "split": "test",
            },
            "model": {
                "explicit_dim": 15,
                "residual_dim": 64,
                "hidden_dim": 32,
                "n_blocks": 2,
                "n_heads": 4,
                "dropout": 0.0,
            },
            "loss": {
                "explicit_dim": 15,
                "lambda_e": 1.0,
                "lambda_z": 1.0,
                "lambda_exist": 0.05,
            },
            "checkpoints": {
                "resume_from": train_result["best_checkpoint_path"],
            },
            "eval": {
                "batch_size": 2,
                "generated_num_samples": 2,
                "output_dir": str(tmp_path / "eval"),
                "num_steps": 4,
            },
            "sampling": {
                "eval_steps": 4,
                "exist_threshold": 0.5,
            },
            "visualization": {
                "enabled": False,
                "generated_num_samples": 2,
            },
            "autodec_decode": {
                "enabled": False,
            },
        }
    )
    eval_result = run_eval_phase2(eval_cfg)
    metrics_path = tmp_path / "eval" / "phase2_smoke_eval" / "metrics.json"
    samples_path = tmp_path / "eval" / "phase2_smoke_eval" / "generated_samples.pt"
    assert eval_result["num_samples"] > 0
    assert metrics_path.is_file()
    assert samples_path.is_file()
