import json
from types import SimpleNamespace


def _cfg(root, checkpoint_path, output_dir, autodec_decode=None):
    return SimpleNamespace(
        seed=0,
        run_name="gendec_eval_debug",
        dataset=SimpleNamespace(root=str(root), split="test"),
        model=SimpleNamespace(token_dim=15, hidden_dim=32, n_blocks=2, n_heads=4, dropout=0.0),
        loss=SimpleNamespace(lambda_exist=0.05, exist_channel=-1),
        optimizer=SimpleNamespace(lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999)),
        trainer=SimpleNamespace(batch_size=2, num_workers=0, num_epochs=1, checkpoint_path=str(checkpoint_path)),
        eval=SimpleNamespace(batch_size=2, output_dir=str(output_dir), generated_num_samples=2, num_steps=4),
        visualization=SimpleNamespace(
            enabled=True,
            root_dir=str(output_dir / "viz"),
            generated_num_samples=10,
            mesh_resolution=8,
            exist_threshold=0.5,
            max_preview_points=128,
        ),
        autodec_decode=autodec_decode,
    )


def _cfg_phase2(root, checkpoint_path, output_dir, residual_dim=4, autodec_decode=None):
    return SimpleNamespace(
        seed=0,
        run_name="gendec_phase2_eval_debug",
        dataset=SimpleNamespace(root=str(root), split="test"),
        model=SimpleNamespace(explicit_dim=15, residual_dim=residual_dim, hidden_dim=32, n_blocks=2, n_heads=4, dropout=0.0),
        loss=SimpleNamespace(explicit_dim=15, lambda_e=1.0, lambda_z=1.0, lambda_exist=0.05, exist_channel=-1),
        optimizer=SimpleNamespace(lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999)),
        trainer=SimpleNamespace(batch_size=2, num_workers=0, num_epochs=1, checkpoint_path=str(checkpoint_path)),
        eval=SimpleNamespace(batch_size=2, output_dir=str(output_dir), generated_num_samples=2, num_steps=4),
        visualization=SimpleNamespace(
            enabled=True,
            root_dir=str(output_dir / "viz"),
            generated_num_samples=10,
            mesh_resolution=8,
            exist_threshold=0.5,
            max_preview_points=128,
        ),
        autodec_decode=autodec_decode,
    )


def test_phase1_evaluator_writes_metrics_and_per_sample_rows(tmp_path):
    from gendec.eval.evaluator import Phase1Evaluator
    from gendec.data.toy_builder import write_toy_teacher_dataset
    from gendec.training.builders import build_dataset, build_loss, build_model
    from gendec.training.checkpoints import save_phase1_checkpoint

    root = tmp_path / "ShapeNet"
    checkpoint_path = tmp_path / "phase1.pt"
    output_dir = tmp_path / "eval"
    write_toy_teacher_dataset(root=root, split="train", num_examples=4, num_points=32)
    write_toy_teacher_dataset(root=root, split="test", num_examples=3, num_points=32)
    cfg = _cfg(root, checkpoint_path, output_dir)

    model = build_model(cfg)
    save_phase1_checkpoint(model, optimizer=None, scheduler=None, epoch=0, loss=0.0, path=checkpoint_path)
    loss_fn = build_loss(cfg)
    dataset = build_dataset(cfg)

    evaluator = Phase1Evaluator(cfg=cfg, model=model, loss_fn=loss_fn, dataset=dataset)
    result = evaluator.evaluate()

    metrics_path = output_dir / "gendec_eval_debug" / "metrics.json"
    per_sample_path = output_dir / "gendec_eval_debug" / "per_sample_metrics.jsonl"
    assert metrics_path.is_file()
    assert per_sample_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["split"] == "test"
    assert metrics["num_samples"] == 3
    assert "generated_active_primitive_count" in metrics["metrics"]
    assert result["metrics"]["num_rows"] == 3

    viz_root = output_dir / "viz" / "gendec_eval_debug" / "test"
    sample_dirs = sorted(viz_root.glob("generated_*"))
    assert len(sample_dirs) == 10
    assert (sample_dirs[0] / "sq_mesh.obj").is_file()
    assert (sample_dirs[0] / "preview_points.ply").is_file()
    assert (sample_dirs[0] / "metadata.json").is_file()


def test_phase1_evaluator_can_decode_generated_scaffolds_with_frozen_autodec_decoder(tmp_path):
    from gendec.eval.evaluator import Phase1Evaluator
    from gendec.tests.eval.test_autodec_bridge import _write_autodec_decoder_assets
    from gendec.data.toy_builder import write_toy_teacher_dataset
    from gendec.training.builders import build_dataset, build_loss, build_model
    from gendec.training.checkpoints import save_phase1_checkpoint

    root = tmp_path / "ShapeNet"
    checkpoint_path = tmp_path / "phase1.pt"
    output_dir = tmp_path / "eval"
    autodec_config_path, autodec_checkpoint_path = _write_autodec_decoder_assets(tmp_path)
    write_toy_teacher_dataset(root=root, split="train", num_examples=4, num_points=32)
    write_toy_teacher_dataset(root=root, split="test", num_examples=3, num_points=32)
    cfg = _cfg(
        root,
        checkpoint_path,
        output_dir,
        autodec_decode=SimpleNamespace(
            enabled=True,
            config_path=str(autodec_config_path),
            checkpoint_path=str(autodec_checkpoint_path),
            reference_limit=2,
            point_count=16,
            f_score_threshold=0.01,
            output_filename="generated_autodec_samples.pt",
        ),
    )

    model = build_model(cfg)
    save_phase1_checkpoint(model, optimizer=None, scheduler=None, epoch=0, loss=0.0, path=checkpoint_path)
    loss_fn = build_loss(cfg)
    dataset = build_dataset(cfg)

    evaluator = Phase1Evaluator(cfg=cfg, model=model, loss_fn=loss_fn, dataset=dataset)
    result = evaluator.evaluate()

    metrics = result["metrics"]
    output_path = output_dir / "gendec_eval_debug" / "generated_autodec_samples.pt"

    assert "coarse_decoded_nn_chamfer_l1" in metrics
    assert "coarse_decoded_active_point_count" in metrics
    assert output_path.is_file()


def test_phase2_evaluator_writes_decoded_point_cloud_visualizations(tmp_path):
    from gendec.data.toy_builder import write_toy_phase2_dataset
    from gendec.eval.evaluator import Phase2Evaluator
    from gendec.tests.eval.test_autodec_bridge import _write_autodec_decoder_assets
    from gendec.training.builders import build_phase2_dataset, build_phase2_loss, build_phase2_model
    from gendec.training.checkpoints import save_phase1_checkpoint

    root = tmp_path / "ShapeNetPhase2"
    checkpoint_path = tmp_path / "phase2.pt"
    output_dir = tmp_path / "eval"
    autodec_config_path, autodec_checkpoint_path = _write_autodec_decoder_assets(tmp_path)
    write_toy_phase2_dataset(root=root, split="train", num_examples=4, num_points=32, residual_dim=4)
    write_toy_phase2_dataset(root=root, split="test", num_examples=3, num_points=32, residual_dim=4)
    cfg = _cfg_phase2(
        root,
        checkpoint_path,
        output_dir,
        residual_dim=4,
        autodec_decode=SimpleNamespace(
            enabled=True,
            config_path=str(autodec_config_path),
            checkpoint_path=str(autodec_checkpoint_path),
            reference_limit=2,
            point_count=16,
            f_score_threshold=0.01,
            output_filename="generated_autodec_samples.pt",
        ),
    )

    model = build_phase2_model(cfg)
    save_phase1_checkpoint(model, optimizer=None, scheduler=None, epoch=0, loss=0.0, path=checkpoint_path)
    loss_fn = build_phase2_loss(cfg)
    dataset = build_phase2_dataset(cfg)

    evaluator = Phase2Evaluator(cfg=cfg, model=model, loss_fn=loss_fn, dataset=dataset)
    result = evaluator.evaluate()

    metrics = result["metrics"]
    output_path = output_dir / "gendec_phase2_eval_debug" / "generated_autodec_samples.pt"
    viz_root = output_dir / "viz" / "gendec_phase2_eval_debug" / "test"
    sample_dirs = sorted(viz_root.glob("generated_*"))

    assert "coarse_decoded_nn_chamfer_l1" in metrics
    assert output_path.is_file()
    assert len(sample_dirs) == 10
    assert (sample_dirs[0] / "sq_mesh.obj").is_file()
    assert (sample_dirs[0] / "preview_points.ply").is_file()
    assert (sample_dirs[0] / "decoded_points.ply").is_file()
