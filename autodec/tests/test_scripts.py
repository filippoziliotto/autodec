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


def test_scripts_folder_has_same_name_documentation():
    doc = Path("autodec/scripts/scripts.md")

    content = doc.read_text()

    assert "run_phase1.sh" in content
    assert "run_phase2.sh" in content
    assert "train_phase1" in content
    assert "train_phase2" in content
