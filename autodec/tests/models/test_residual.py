import torch


def test_part_residual_projector_pools_point_features_by_assignment():
    from autodec.models.residual import PartResidualProjector

    projector = PartResidualProjector(feature_dim=2, residual_dim=3, hidden_dim=4)
    point_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    assign_matrix = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])

    pooled = projector.pool_point_features(point_features, assign_matrix)

    expected = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]])
    assert torch.allclose(pooled, expected)


def test_part_residual_projector_builds_mean_max_variance_statistics():
    from autodec.models.residual import PartResidualProjector

    projector = PartResidualProjector(feature_dim=2, residual_dim=3, hidden_dim=4)
    point_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    assign_matrix = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])

    stats = projector.pool_point_feature_stats(point_features, assign_matrix)

    expected_mean = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]])
    expected_max = torch.tensor([[[5.0, 6.0], [3.0, 4.0]]])
    expected_var = torch.tensor([[[4.0, 4.0], [0.0, 0.0]]])
    expected = torch.cat([expected_mean, expected_max, expected_var], dim=-1)
    assert torch.allclose(stats, expected)


def test_part_residual_projector_max_ignores_inactive_assignments():
    from autodec.models.residual import PartResidualProjector

    projector = PartResidualProjector(feature_dim=2, residual_dim=3, hidden_dim=4)
    point_features = torch.tensor([[[-5.0, -2.0], [-7.0, -8.0], [9.0, 10.0]]])
    assign_matrix = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

    stats = projector.pool_point_feature_stats(point_features, assign_matrix)
    max_features = stats[..., 2:4]

    expected = torch.tensor([[[-5.0, -2.0], [9.0, 10.0]]])
    assert torch.allclose(max_features, expected)


def test_part_residual_projector_returns_residual_and_pooled_features():
    from autodec.models.residual import PartResidualProjector

    projector = PartResidualProjector(feature_dim=4, residual_dim=5, hidden_dim=8)
    sq_features = torch.randn(2, 3, 4)
    point_features = torch.randn(2, 7, 4)
    assign_matrix = torch.softmax(torch.randn(2, 7, 3), dim=-1)

    residual, pooled = projector(
        sq_features,
        point_features,
        assign_matrix,
        return_pooled=True,
    )

    assert residual.shape == (2, 3, 5)
    assert pooled.shape == (2, 3, 4)
    assert projector.net[0].in_features == 16
