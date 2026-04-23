from pathlib import Path


NORMALIZATION_FILENAME = "normalization.pt"
SCAFFOLD_FILENAME = "teacher_scaffold.pt"


def normalization_stats_path(root):
    return Path(root) / NORMALIZATION_FILENAME


def model_dir(root, category_id, model_id):
    return Path(root) / str(category_id) / str(model_id)


def scaffold_example_path(root, category_id, model_id):
    return model_dir(root, category_id, model_id) / SCAFFOLD_FILENAME


def split_manifest_path(root, category_id, split):
    return Path(root) / str(category_id) / f"{split}.lst"


def _read_manifest(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def available_categories(root, categories=None):
    root = Path(root)
    if categories is None:
        categories = [path.name for path in root.iterdir() if path.is_dir()]
    return sorted(str(category_id) for category_id in categories if (root / str(category_id)).is_dir())


def build_category_vocab(root, categories=None):
    category_ids = available_categories(root, categories=categories)
    return category_ids, {category_id: index for index, category_id in enumerate(category_ids)}


def iter_exported_examples(root, split=None, categories=None):
    root = Path(root)
    available = available_categories(root, categories=categories)

    for category_id in available:
        category_dir = root / category_id
        if not category_dir.is_dir():
            continue

        manifest_path = split_manifest_path(root, category_id, split) if split is not None else None
        if manifest_path is not None and manifest_path.is_file():
            model_ids = _read_manifest(manifest_path)
        else:
            model_ids = sorted(path.name for path in category_dir.iterdir() if path.is_dir())

        for model_id in model_ids:
            path = scaffold_example_path(root, category_id, model_id)
            if path.is_file():
                yield {
                    "category_id": category_id,
                    "model_id": model_id,
                    "path": path,
                }


def write_split_manifest(root, split, model_index):
    grouped = {}
    for item in model_index:
        grouped.setdefault(str(item["category_id"]), []).append(str(item["model_id"]))

    for category_id, model_ids in grouped.items():
        manifest_path = split_manifest_path(root, category_id, split)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("\n".join(model_ids) + "\n", encoding="utf-8")
