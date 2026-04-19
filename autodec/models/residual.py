import torch
import torch.nn as nn


class PartResidualProjector(nn.Module):
    """Build per-primitive residual tokens from SQ and local point features."""

    def __init__(self, feature_dim=128, residual_dim=64, hidden_dim=None, eps=1e-6):
        super().__init__()
        self.feature_dim = feature_dim
        self.residual_dim = residual_dim
        self.hidden_dim = hidden_dim or feature_dim
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, residual_dim),
        )

    def pool_point_features(self, point_features, assign_matrix):
        mass = assign_matrix.sum(dim=1).clamp_min(self.eps)
        pooled = torch.einsum("bnp,bnh->bph", assign_matrix, point_features)
        return pooled / mass.unsqueeze(-1)

    def pool_point_feature_stats(self, point_features, assign_matrix):
        mean = self.pool_point_features(point_features, assign_matrix)
        weighted_features = assign_matrix.unsqueeze(-1) * point_features.unsqueeze(2)
        inactive = assign_matrix <= self.eps
        weighted_features = weighted_features.masked_fill(inactive.unsqueeze(-1), -torch.inf)
        max_features = weighted_features.max(dim=1).values
        max_features = torch.where(
            torch.isfinite(max_features),
            max_features,
            torch.zeros_like(max_features),
        )
        diff = point_features.unsqueeze(2) - mean.unsqueeze(1)
        mass = assign_matrix.sum(dim=1).clamp_min(self.eps)
        var = (assign_matrix.unsqueeze(-1) * diff.square()).sum(dim=1)
        var = var / mass.unsqueeze(-1)
        return torch.cat([mean, max_features, var], dim=-1)

    def forward(
        self,
        sq_features,
        point_features,
        assign_matrix,
        return_pooled=False,
    ):
        pooled = self.pool_point_features(point_features, assign_matrix)
        pooled_stats = self.pool_point_feature_stats(point_features, assign_matrix)
        residual_input = torch.cat([sq_features, pooled_stats], dim=-1)
        residual = self.net(residual_input)
        if return_pooled:
            return residual, pooled
        return residual
