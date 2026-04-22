import torch


def _outdict():
    return {
        "decoded_points": torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [100.0, 0.0, 0.0],
                    [101.0, 0.0, 0.0],
                ]
            ]
        ),
        "part_ids": torch.tensor([0, 0, 1, 1]),
        "exist": torch.tensor([[[0.9], [0.1]]]),
    }


def test_prune_decoded_points_keeps_points_from_active_primitives():
    from autodec.utils.inference import prune_decoded_points

    pruned = prune_decoded_points(_outdict(), exist_threshold=0.5)

    assert len(pruned) == 1
    assert torch.equal(
        pruned[0],
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )


def test_prune_decoded_points_can_resample_to_fixed_count():
    from autodec.utils.inference import prune_decoded_points

    pruned = prune_decoded_points(_outdict(), exist_threshold=0.5, target_count=4)

    assert pruned.shape == (1, 4, 3)
    assert torch.equal(
        pruned,
        torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ]
        ),
    )


def test_prune_decoded_points_falls_back_to_highest_existence_primitive():
    from autodec.utils.inference import prune_decoded_points

    outdict = _outdict()
    outdict["exist"] = torch.tensor([[[0.4], [0.3]]])

    pruned = prune_decoded_points(outdict, exist_threshold=0.5)

    assert torch.equal(
        pruned[0],
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )


def test_prune_points_by_active_primitives_supports_surface_points():
    from autodec.utils.inference import prune_points_by_active_primitives

    outdict = _outdict()
    outdict["surface_points"] = outdict["decoded_points"] + 10.0

    pruned = prune_points_by_active_primitives(
        outdict,
        "surface_points",
        exist_threshold=0.5,
    )

    assert torch.equal(
        pruned[0],
        torch.tensor(
            [
                [10.0, 10.0, 10.0],
                [11.0, 10.0, 10.0],
            ]
        ),
    )
