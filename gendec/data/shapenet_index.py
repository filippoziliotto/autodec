from pathlib import Path


def _read_manifest(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scan_source_shapenet_models(dataset_path, categories=None, split=None, require_manifest=False):
    dataset_path = Path(dataset_path)
    if categories is None:
        categories = sorted(path.name for path in dataset_path.iterdir() if path.is_dir())

    models = []
    for category_id in categories:
        category_dir = dataset_path / category_id
        if not category_dir.is_dir():
            continue

        manifest_path = category_dir / f"{split}.lst" if split is not None else None
        if manifest_path is not None and manifest_path.is_file():
            model_ids = _read_manifest(manifest_path)
        elif manifest_path is not None and require_manifest:
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
        else:
            model_ids = sorted(
                path.name
                for path in category_dir.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            )

        for model_id in model_ids:
            model_dir = category_dir / model_id
            if model_dir.is_dir():
                models.append(
                    {
                        "category_id": category_id,
                        "model_id": model_id,
                        "model_dir": model_dir,
                    }
                )

    return models
