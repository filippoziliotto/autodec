from collections import defaultdict
from dataclasses import dataclass
from math import ceil


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
    num_samples=20,
    min_categories=5,
    samples_per_category=4,
    categories=None,
):
    """Select deterministic ShapeNet test samples balanced across categories."""

    num_samples = int(num_samples)
    min_categories = int(min_categories)
    samples_per_category = int(samples_per_category)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if min_categories <= 0:
        raise ValueError("min_categories must be positive")
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

    target_category_count = max(min_categories, ceil(num_samples / samples_per_category))
    available_categories = sorted(grouped)
    if len(available_categories) < target_category_count:
        raise ValueError(
            f"Need at least {target_category_count} categories with test samples, "
            f"found {len(available_categories)}"
        )

    selected = []
    remaining = num_samples
    for category in available_categories[:target_category_count]:
        need = min(samples_per_category, remaining)
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
        remaining -= need
        if remaining <= 0:
            break

    return sorted(selected, key=lambda item: item.dataset_index)

