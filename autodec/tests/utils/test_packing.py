import torch


def _outdict(batch=2, primitives=3):
    rotate = torch.eye(3).view(1, 1, 3, 3).repeat(batch, primitives, 1, 1)
    return {
        "scale": torch.ones(batch, primitives, 3),
        "shape": torch.ones(batch, primitives, 2) * 0.5,
        "trans": torch.arange(batch * primitives * 3, dtype=torch.float32).view(
            batch, primitives, 3
        ),
        "rotate": rotate,
        "exist_logit": torch.ones(batch, primitives, 1) * 2.0,
        "exist": torch.ones(batch, primitives, 1) * 0.25,
        "rotation_quat": torch.ones(batch, primitives, 4),
    }


def test_pack_decoder_primitive_features_uses_matrix_rotation_and_logit():
    from autodec.utils.packing import pack_decoder_primitive_features

    packed = pack_decoder_primitive_features(_outdict())

    assert packed.shape == (2, 3, 18)
    assert torch.allclose(packed[..., -1:], torch.ones(2, 3, 1) * 2.0)


def test_pack_serialized_primitive_features_uses_quaternion_when_available():
    from autodec.utils.packing import pack_serialized_primitive_features

    packed = pack_serialized_primitive_features(_outdict(), rotation_mode="quat")

    assert packed.shape == (2, 3, 13)


def test_repeat_by_part_ids_repeats_slots_per_surface_point():
    from autodec.utils.packing import repeat_by_part_ids

    values = torch.arange(2 * 3 * 4).view(2, 3, 4)
    part_ids = torch.tensor([0, 2, 1, 2])

    repeated = repeat_by_part_ids(values, part_ids)

    assert repeated.shape == (2, 4, 4)
    assert torch.equal(repeated[:, 0], values[:, 0])
    assert torch.equal(repeated[:, 1], values[:, 2])
    assert torch.equal(repeated[:, 2], values[:, 1])
