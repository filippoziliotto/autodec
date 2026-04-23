import torch


def compute_assignment_mass(assign_matrix):
    return assign_matrix.to(torch.float32).mean(dim=0)


def compute_primitive_volume(scale):
    return scale.to(torch.float32).prod(dim=-1)


def deterministic_sort_indices(exist, mass, volume, translation):
    exist = exist.squeeze(-1).to(torch.float32)
    mass = mass.to(torch.float32)
    volume = volume.to(torch.float32)
    translation_x = translation[..., 0].to(torch.float32)
    order = list(range(exist.shape[0]))
    order.sort(
        key=lambda idx: (
            -float(exist[idx]),
            -float(mass[idx]),
            -float(volume[idx]),
            float(translation_x[idx]),
        )
    )
    return torch.tensor(order, dtype=torch.long)


def reorder_teacher_outputs(payload, order):
    reordered = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            reordered[key] = value
            continue
        if key == "assign_matrix":
            reordered[key] = value[..., order]
        elif value.ndim >= 1 and value.shape[0] == order.shape[0]:
            reordered[key] = value.index_select(0, order)
        else:
            reordered[key] = value
    return reordered
