import json
import math
from pathlib import Path


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class EpochMetricLogger:
    """Append per-epoch training metrics as JSON lines."""

    def __init__(self, path, append=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            self.path.write_text("", encoding="utf-8")

    def write(self, row):
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(_jsonable(row), handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
