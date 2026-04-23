import sys


def test_explicit_config_argument_reads_separate_value(monkeypatch):
    from gendec.config import explicit_config_argument

    monkeypatch.setattr(sys, "argv", ["export_teacher.py", "--config", "gendec/configs/teacher_export.yaml"])

    assert explicit_config_argument("teacher_export.yaml") == "gendec/configs/teacher_export.yaml"


def test_explicit_config_argument_reads_equals_value(monkeypatch):
    from gendec.config import explicit_config_argument

    monkeypatch.setattr(sys, "argv", ["train.py", "--config=gendec/configs/train.yaml"])

    assert explicit_config_argument("train.yaml") == "gendec/configs/train.yaml"


def test_explicit_config_argument_ignores_hydra_style_flags(monkeypatch):
    from gendec.config import explicit_config_argument

    monkeypatch.setattr(sys, "argv", ["export_teacher.py", "--config-name", "teacher_export"])

    assert explicit_config_argument("teacher_export.yaml") is None
