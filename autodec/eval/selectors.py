from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectedSample:
    dataset_index: int
    category: str
    model_id: str


def _dataset_models(dataset):
    models = getattr(dataset, "models", None)
    if models is None:
        raise ValueError("Category-balanced evaluation requires dataset.models metadata")
    return models


def select_category_balanced_indices(
    dataset,
    samples_per_category=2,
    categories=None,
):
    """Select deterministic ShapeNet test samples from every category."""

    samples_per_category = int(samples_per_category)
    if samples_per_category <= 0:
        raise ValueError("samples_per_category must be positive")

    requested_categories = set(categories) if categories is not None else None
    grouped = defaultdict(list)
    for dataset_index, model in enumerate(_dataset_models(dataset)):
        category = model.get("category")
        if category is None:
            continue
        if requested_categories is not None and category not in requested_categories:
            continue
        grouped[category].append((dataset_index, model.get("model_id", str(dataset_index))))

    available_categories = sorted(grouped)
    if not available_categories:
        raise ValueError("No categories with test samples found")

    selected = []
    for category in available_categories:
        need = samples_per_category
        candidates = grouped[category]
        if len(candidates) < need:
            raise ValueError(
                f"Category {category} has {len(candidates)} test samples, need {need}"
            )
        for dataset_index, model_id in candidates[:need]:
            selected.append(
                SelectedSample(
                    dataset_index=dataset_index,
                    category=category,
                    model_id=model_id,
                )
            )

    return sorted(selected, key=lambda item: item.dataset_index)
