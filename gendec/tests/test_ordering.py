import torch


def test_deterministic_sort_indices_follow_priority_rules():
    from gendec.data.ordering import deterministic_sort_indices

    exist = torch.tensor([[0.8], [0.8], [0.5], [0.8]])
    mass = torch.tensor([0.2, 0.4, 0.9, 0.4])
    volume = torch.tensor([0.3, 0.2, 0.1, 0.1])
    translation = torch.tensor(
        [
            [0.3, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ]
    )

    order = deterministic_sort_indices(exist, mass, volume, translation)

    assert order.tolist() == [1, 3, 0, 2]


def test_reorder_teacher_outputs_reorders_assignments_and_tokens_together():
    from gendec.data.ordering import reorder_teacher_outputs

    payload = {
        "scale": torch.tensor([[1.0], [2.0], [3.0]]),
        "shape": torch.tensor([[11.0], [12.0], [13.0]]),
        "rot6d": torch.tensor([[21.0], [22.0], [23.0]]),
        "trans": torch.tensor([[31.0], [32.0], [33.0]]),
        "exist_logit": torch.tensor([[41.0], [42.0], [43.0]]),
        "exist": torch.tensor([[0.1], [0.2], [0.3]]),
        "mass": torch.tensor([0.4, 0.5, 0.6]),
        "volume": torch.tensor([0.7, 0.8, 0.9]),
        "assign_matrix": torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    }

    reordered = reorder_teacher_outputs(payload, torch.tensor([2, 0, 1]))

    assert reordered["scale"].squeeze(-1).tolist() == [3.0, 1.0, 2.0]
    assert reordered["assign_matrix"].tolist() == [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]
