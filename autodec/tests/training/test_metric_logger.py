import json


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_epoch_metric_logger_creates_parent_writes_jsonl_and_flushes(tmp_path):
    from autodec.training.metric_logger import EpochMetricLogger

    path = tmp_path / "run" / "metrics.jsonl"
    logger = EpochMetricLogger(path, append=False)

    logger.write({"epoch": 1, "train": {"all": 0.5}})

    assert path.exists()
    assert _read_rows(path) == [{"epoch": 1, "train": {"all": 0.5}}]


def test_epoch_metric_logger_append_and_overwrite_modes(tmp_path):
    from autodec.training.metric_logger import EpochMetricLogger

    path = tmp_path / "metrics.jsonl"
    EpochMetricLogger(path, append=False).write({"epoch": 1})
    EpochMetricLogger(path, append=True).write({"epoch": 2})

    assert _read_rows(path) == [{"epoch": 1}, {"epoch": 2}]

    EpochMetricLogger(path, append=False).write({"epoch": 3})

    assert _read_rows(path) == [{"epoch": 3}]
