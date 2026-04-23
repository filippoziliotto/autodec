STANDARD_SHAPENET_SPLITS = ("train", "val", "test")


def resolve_split_names(split=None, splits=None, default="train"):
    if splits is None:
        splits = [default if split is None else split]
    elif isinstance(splits, str):
        splits = [splits]
    else:
        splits = list(splits)

    resolved = []
    for item in splits:
        if item is None:
            name = None
        else:
            name = str(item).strip().lower()
            if name == "all":
                resolved.extend(STANDARD_SHAPENET_SPLITS)
                continue
        resolved.append(name)

    unique = []
    for name in resolved:
        if name not in unique:
            unique.append(name)

    ordered = []
    for name in STANDARD_SHAPENET_SPLITS:
        if name in unique:
            ordered.append(name)
    for name in unique:
        if name not in ordered:
            ordered.append(name)
    return ordered
