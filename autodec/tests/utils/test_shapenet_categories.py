from types import SimpleNamespace


def test_apply_category_split_defaults_to_all_when_missing():
    from autodec.utils.shapenet_categories import ALL_SHAPENET_CATEGORIES, apply_shapenet_category_split

    cfg = SimpleNamespace(shapenet=SimpleNamespace(categories=["03001627"]))

    apply_shapenet_category_split(cfg)

    assert cfg.shapenet.categories == ALL_SHAPENET_CATEGORIES


def test_apply_category_split_keeps_explicit_categories_when_null():
    from autodec.utils.shapenet_categories import apply_shapenet_category_split

    cfg = SimpleNamespace(shapenet=SimpleNamespace(category_split=None, categories=["03001627"]))

    apply_shapenet_category_split(cfg)

    assert cfg.shapenet.categories == ["03001627"]


def test_apply_category_split_uses_paper_seen_categories():
    from autodec.utils.shapenet_categories import PAPER_SEEN_CATEGORIES, apply_shapenet_category_split

    cfg = SimpleNamespace(shapenet=SimpleNamespace(category_split="paper_seen", categories=None))

    apply_shapenet_category_split(cfg)

    assert cfg.shapenet.categories == PAPER_SEEN_CATEGORIES


def test_apply_category_split_uses_paper_unseen_categories():
    from autodec.utils.shapenet_categories import PAPER_UNSEEN_CATEGORIES, apply_shapenet_category_split

    cfg = SimpleNamespace(shapenet=SimpleNamespace(category_split="paper_unseen", categories=None))

    apply_shapenet_category_split(cfg)

    assert cfg.shapenet.categories == PAPER_UNSEEN_CATEGORIES


def test_apply_category_split_updates_omegaconf_dictconfig():
    import pytest

    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    from autodec.utils.shapenet_categories import ALL_SHAPENET_CATEGORIES, apply_shapenet_category_split

    cfg = OmegaConf.create({"shapenet": {"category_split": "all", "categories": ["03001627"]}})

    apply_shapenet_category_split(cfg)

    assert list(cfg.shapenet.categories) == ALL_SHAPENET_CATEGORIES


def test_apply_category_split_rejects_unknown_split():
    import pytest

    from autodec.utils.shapenet_categories import apply_shapenet_category_split

    cfg = SimpleNamespace(shapenet=SimpleNamespace(category_split="bad", categories=None))

    with pytest.raises(ValueError, match="Unsupported ShapeNet category_split"):
        apply_shapenet_category_split(cfg)
