from pathlib import Path


def test_phase_scripts_call_autodec_training_entrypoint():
    scripts_dir = Path("autodec/scripts")
    expected = {
        "run_smoke.sh": "smoke",
        "run_phase1.sh": "train_phase1",
        "run_phase2.sh": "train_phase2",
    }

    for filename, config_name in expected.items():
        script = scripts_dir / filename
        content = script.read_text()

        assert content.startswith("#!/usr/bin/env bash")
        assert "python -m autodec.training.train" in content
        assert f"--config-name {config_name}" in content
        assert '"$@"' in content


def test_eval_script_calls_autodec_eval_entrypoint():
    script = Path("autodec/scripts/run_eval_test.sh")
    content = script.read_text()

    assert content.startswith("#!/usr/bin/env bash")
    assert "ensure_fast_sampler" in content
    assert "python -m autodec.eval.run" in content
    assert "--config-name eval_test" in content
    assert '"$@"' in content


def test_multigpu_pipeline_runs_phase_training_and_eval():
    script = Path("autodec/scripts/run_multigpu_pipeline.sh")
    content = script.read_text()

    assert content.startswith("#!/usr/bin/env bash")
    assert "ensure_fast_sampler" in content
    assert "torchrun --nproc_per_node=\"${NUM_GPUS}\" -m autodec.training.train" in content
    assert "--config-name train_phase1" in content
    assert "--config-name train_phase2" in content
    assert "trainer.num_epochs=\"${PHASE1_EPOCHS}\"" in content
    assert "trainer.num_epochs=\"${PHASE2_EPOCHS}\"" in content
    assert "trainer.batch_size=\"${BATCH_SIZE_PER_GPU}\"" in content
    assert 'PHASE1_CKPT="${PHASE1_CKPT:-${CHECKPOINT_ROOT}/${PHASE1_RUN_NAME}/best.pt}"' in content
    assert 'PHASE2_CKPT="${PHASE2_CKPT:-${CHECKPOINT_ROOT}/${PHASE2_RUN_NAME}/best.pt}"' in content
    assert "checkpoints.resume_from=\"${PHASE1_CKPT}\"" in content
    assert "python -m autodec.eval.run" in content
    assert "checkpoints.resume_from=\"${PHASE2_CKPT}\"" in content
    assert '"$@"' in content


def test_scripts_folder_has_same_name_documentation():
    doc = Path("autodec/scripts/scripts.md")

    content = doc.read_text()

    assert "run_phase1.sh" in content
    assert "run_phase2.sh" in content
    assert "run_eval_test.sh" in content
    assert "run_multigpu_pipeline.sh" in content
    assert "train_phase1" in content
    assert "train_phase2" in content
    assert "eval_test" in content
