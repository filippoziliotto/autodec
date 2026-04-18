import pytest


class FakeShapeNet:
    def __init__(self, counts_by_category):
        self.models = []
        for category, count in counts_by_category.items():
            for item_index in range(count):
                self.models.append(
                    {
                        "category": category,
                        "model_id": f"{category}_{item_index:04d}",
                    }
                )


def test_category_balanced_indices_returns_twenty_samples_from_five_categories():
    from autodec.eval.selectors import select_category_balanced_indices

    dataset = FakeShapeNet(
        {
            "03001627": 10,
            "02691156": 10,
            "04379243": 10,
            "02958343": 10,
            "04256520": 10,
            "04530566": 10,
        }
    )

    selection = select_category_balanced_indices(
        dataset,
        num_samples=20,
        min_categories=5,
        samples_per_category=4,
    )

    assert len(selection) == 20
    assert sorted({item.category for item in selection}) == [
        "02691156",
        "02958343",
        "03001627",
        "04256520",
        "04379243",
    ]
    counts = {category: 0 for category in {item.category for item in selection}}
    for item in selection:
        counts[item.category] += 1
    assert set(counts.values()) == {4}
    assert [item.dataset_index for item in selection] == sorted(
        item.dataset_index for item in selection
    )


def test_category_balanced_indices_fails_when_too_few_categories_exist():
    from autodec.eval.selectors import select_category_balanced_indices

    dataset = FakeShapeNet({"02691156": 10, "03001627": 10})

    with pytest.raises(ValueError, match="at least 5 categories"):
        select_category_balanced_indices(
            dataset,
            num_samples=20,
            min_categories=5,
            samples_per_category=4,
        )

