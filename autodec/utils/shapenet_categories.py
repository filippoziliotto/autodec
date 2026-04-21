"""ShapeNet category split helpers for AutoDec experiments."""

ALL_SHAPENET_CATEGORIES = [
    "02691156",  # airplane
    "02828884",  # bench
    "03001627",  # chair
    "03636649",  # lamp
    "04090263",  # rifle
    "04379243",  # table
    "02958343",  # car
    "04256520",  # sofa
    "03691459",  # loudspeaker
    "02933112",  # cabinet
    "03211117",  # display
    "04401088",  # telephone
    "04530566",  # watercraft
]

PAPER_SEEN_CATEGORIES = [
    "02691156",  # airplane
    "02828884",  # bench
    "03001627",  # chair
    "03636649",  # lamp
    "04090263",  # rifle
    "04379243",  # table
]

PAPER_UNSEEN_CATEGORIES = [
    "02958343",  # car
    "04256520",  # sofa
    "03691459",  # loudspeaker
    "02933112",  # cabinet
    "03211117",  # display
    "04401088",  # telephone
    "04530566",  # watercraft
]

CATEGORY_SPLITS = {
    "all": ALL_SHAPENET_CATEGORIES,
    "paper_seen": PAPER_SEEN_CATEGORIES,
    "paper_unseen": PAPER_UNSEEN_CATEGORIES,
}


def _get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set_attr(obj, name, value):
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def apply_shapenet_category_split(cfg):
    """Resolve cfg.shapenet.category_split into cfg.shapenet.categories.

    `category_split: null` preserves the explicit `categories` value for
    debugging or single-category runs. Missing `category_split` defaults to the
    paper in-category setup over all 13 ShapeNet classes.
    """

    shapenet_cfg = _get_attr(cfg, "shapenet")
    if shapenet_cfg is None:
        return cfg

    category_split = _get_attr(shapenet_cfg, "category_split", "all")
    if category_split is None:
        return cfg

    category_split = str(category_split)
    if category_split not in CATEGORY_SPLITS:
        valid = ", ".join(sorted(CATEGORY_SPLITS))
        raise ValueError(
            f"Unsupported ShapeNet category_split={category_split!r}. "
            f"Expected one of: {valid}, or null to use shapenet.categories."
        )

    _set_attr(shapenet_cfg, "categories", list(CATEGORY_SPLITS[category_split]))
    return cfg
