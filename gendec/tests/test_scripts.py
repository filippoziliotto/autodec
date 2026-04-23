from pathlib import Path


def test_gendec_scripts_exist():
    root = Path("gendec/scripts")
    expected = [
        root / "common.sh",
        root / "export_shapenet_teacher.sh",
        root / "export_shapenet_phase2_teacher.sh",
        root / "train_phase1.sh",
        root / "train_phase2.sh",
        root / "eval_val.sh",
        root / "eval_test.sh",
        root / "eval_phase2_val.sh",
        root / "eval_phase2_test.sh",
    ]

    missing = [str(path) for path in expected if not path.is_file()]
    assert not missing, f"Missing scripts: {missing}"
