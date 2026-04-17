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
            nn.Linear(feature_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, residual_dim),
        )

    def pool_point_features(self, point_features, assign_matrix):
        mass = assign_matrix.sum(dim=1).clamp_min(self.eps)
        pooled = torch.einsum("bnp,bnh->bph", assign_matrix, point_features)
        return pooled / mass.unsqueeze(-1)

    def forward(
        self,
        sq_features,
        point_features,
        assign_matrix,
        return_pooled=False,
    ):
        pooled = self.pool_point_features(point_features, assign_matrix)
        residual_input = torch.cat([sq_features, pooled], dim=-1)
        residual = self.net(residual_input)
        if return_pooled:
            return residual, pooled
        return residual
