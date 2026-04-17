import torch


def _exist_feature(outdict):
    if "exist_logit" in outdict:
        return outdict["exist_logit"]
    exist = outdict["exist"].clamp(1e-6, 1 - 1e-6)
    return torch.logit(exist)


def pack_decoder_primitive_features(outdict):
    """Pack primitive features used by the neural decoder.

    The decoder consumes the rotation matrix returned by the current SuperDec
    heads. This is intentionally separate from compact serialized features.
    """

    rotate = outdict["rotate"].reshape(*outdict["rotate"].shape[:2], 9)
    return torch.cat(
        [
            outdict["scale"],
            outdict["shape"],
            outdict["trans"],
            rotate,
            _exist_feature(outdict),
        ],
        dim=-1,
    )


def _matrix_to_quaternion(rotation):
    batch_shape = rotation.shape[:-2]
    matrix = rotation.reshape(-1, 3, 3)
    quat = matrix.new_empty(matrix.shape[0], 4)

    trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    positive = trace > 0

    s = torch.sqrt(torch.clamp(trace[positive] + 1.0, min=1e-12)) * 2.0
    quat[positive, 0] = 0.25 * s
    quat[positive, 1] = (matrix[positive, 2, 1] - matrix[positive, 1, 2]) / s
    quat[positive, 2] = (matrix[positive, 0, 2] - matrix[positive, 2, 0]) / s
    quat[positive, 3] = (matrix[positive, 1, 0] - matrix[positive, 0, 1]) / s

    remaining = ~positive
    if remaining.any():
        rem = matrix[remaining]
        diag = torch.stack([rem[:, 0, 0], rem[:, 1, 1], rem[:, 2, 2]], dim=-1)
        idx = diag.argmax(dim=-1)
        quat_rem = rem.new_empty(rem.shape[0], 4)

        x_case = idx == 0
        if x_case.any():
            m = rem[x_case]
            sx = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-12)) * 2.0
            quat_rem[x_case, 0] = (m[:, 2, 1] - m[:, 1, 2]) / sx
            quat_rem[x_case, 1] = 0.25 * sx
            quat_rem[x_case, 2] = (m[:, 0, 1] + m[:, 1, 0]) / sx
            quat_rem[x_case, 3] = (m[:, 0, 2] + m[:, 2, 0]) / sx

        y_case = idx == 1
        if y_case.any():
            m = rem[y_case]
            sy = torch.sqrt(torch.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-12)) * 2.0
            quat_rem[y_case, 0] = (m[:, 0, 2] - m[:, 2, 0]) / sy
            quat_rem[y_case, 1] = (m[:, 0, 1] + m[:, 1, 0]) / sy
            quat_rem[y_case, 2] = 0.25 * sy
            quat_rem[y_case, 3] = (m[:, 1, 2] + m[:, 2, 1]) / sy

        z_case = idx == 2
        if z_case.any():
            m = rem[z_case]
            sz = torch.sqrt(torch.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-12)) * 2.0
            quat_rem[z_case, 0] = (m[:, 1, 0] - m[:, 0, 1]) / sz
            quat_rem[z_case, 1] = (m[:, 0, 2] + m[:, 2, 0]) / sz
            quat_rem[z_case, 2] = (m[:, 1, 2] + m[:, 2, 1]) / sz
            quat_rem[z_case, 3] = 0.25 * sz

        quat[remaining] = quat_rem

    return torch.nn.functional.normalize(quat.reshape(*batch_shape, 4), dim=-1)


def pack_serialized_primitive_features(outdict, rotation_mode="quat"):
    """Pack compact primitive features for reporting/storage."""

    if rotation_mode == "quat":
        rotation = outdict.get("rotation_quat")
        if rotation is None:
            rotation = _matrix_to_quaternion(outdict["rotate"])
    elif rotation_mode == "6d":
        rotation = outdict.get("rotation_6d")
        if rotation is None:
            rotation = outdict["rotate"][..., :2].reshape(*outdict["rotate"].shape[:2], 6)
    else:
        raise ValueError(f"Unsupported rotation_mode: {rotation_mode}")

    return torch.cat(
        [
            outdict["scale"],
            outdict["shape"],
            outdict["trans"],
            rotation,
            _exist_feature(outdict),
        ],
        dim=-1,
    )


def repeat_by_part_ids(values, part_ids):
    return values[:, part_ids, :]
