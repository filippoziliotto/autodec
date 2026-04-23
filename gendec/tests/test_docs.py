from pathlib import Path


def test_source_directories_have_matching_folder_docs():
    root = Path("gendec")
    expected = [
        root / "gendec.md",
        root / "configs" / "configs.md",
        root / "data" / "data.md",
        root / "docs" / "docs.md",
        root / "eval" / "eval.md",
        root / "losses" / "losses.md",
        root / "models" / "models.md",
        root / "scripts" / "scripts.md",
        root / "tests" / "tests.md",
        root / "tests" / "eval" / "eval.md",
        root / "training" / "training.md",
        root / "utils" / "utils.md",
    ]

    missing = [str(path) for path in expected if not path.is_file()]

    assert not missing, f"Missing folder docs: {missing}"
