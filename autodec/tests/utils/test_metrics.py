import torch


def test_offset_ratio_uses_raw_offsets_against_scaffold_norm():
    from autodec.utils.metrics import offset_ratio

    surface_points = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]])
    offsets = torch.tensor([[[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]])

    assert torch.allclose(offset_ratio(surface_points, offsets), torch.tensor(1.0))


def test_active_counts_and_assignment_entropy():
    from autodec.utils.metrics import active_decoded_point_count, active_primitive_count, primitive_mass_entropy

    exist = torch.tensor([[[0.9], [0.1], [0.8]]])
    weights = torch.tensor([[0.9, 0.9, 0.1, 0.8]])
    assign_matrix = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    assert torch.allclose(active_primitive_count(exist), torch.tensor(2.0))
    assert torch.allclose(active_decoded_point_count(weights), torch.tensor(3.0))
    assert primitive_mass_entropy(assign_matrix).item() == 0.0


def test_scaffold_vs_decoded_chamfer_reports_both_values():
    from autodec.utils.metrics import scaffold_vs_decoded_chamfer

    target = torch.tensor([[[0.0, 0.0, 0.0]]])
    surface = torch.tensor([[[1.0, 0.0, 0.0]]])
    decoded = torch.tensor([[[0.0, 0.0, 0.0]]])
    weights = torch.ones(1, 1)

    metrics = scaffold_vs_decoded_chamfer(surface, decoded, target, weights)

    assert metrics["decoded_chamfer"] < metrics["scaffold_chamfer"]
