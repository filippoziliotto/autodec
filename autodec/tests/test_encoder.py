from types import SimpleNamespace

import torch
from torch import nn


class FakePointEncoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, points):
        batch, n_points, _ = points.shape
        values = torch.arange(
            batch * n_points * self.feature_dim,
            dtype=points.dtype,
            device=points.device,
        )
        return values.view(batch, n_points, self.feature_dim)


class FakeLayers(nn.Module):
    def __init__(self, n_queries, feature_dim):
        super().__init__()
        self.n_queries = n_queries
        self.feature_dim = feature_dim
        self.project_queries = nn.Identity()

    def forward(self, init_queries, point_features):
        batch, n_points, _ = point_features.shape
        queries = torch.ones(
            batch,
            self.n_queries + 1,
            self.feature_dim,
            dtype=point_features.dtype,
            device=point_features.device,
        )
        assign_logits = torch.zeros(
            batch,
            n_points,
            self.n_queries,
            dtype=point_features.dtype,
            device=point_features.device,
        )
        assign_logits[:, :, 0] = 2.0
        return [queries], [assign_logits]


class FakeHeads(nn.Module):
    def forward(self, sq_features):
        batch, n_queries, _ = sq_features.shape
        device = sq_features.device
        dtype = sq_features.dtype
        exist_logit = torch.zeros(batch, n_queries, 1, device=device, dtype=dtype)
        return {
            "scale": torch.ones(batch, n_queries, 3, device=device, dtype=dtype),
            "shape": torch.ones(batch, n_queries, 2, device=device, dtype=dtype),
            "rotate": torch.eye(3, device=device, dtype=dtype)
            .view(1, 1, 3, 3)
            .repeat(batch, n_queries, 1, 1),
            "trans": torch.zeros(batch, n_queries, 3, device=device, dtype=dtype),
            "exist_logit": exist_logit,
            "exist": torch.sigmoid(exist_logit),
        }


def _ctx(feature_dim=4, n_queries=2, residual_dim=3):
    return SimpleNamespace(
        residual_dim=residual_dim,
        decoder=SimpleNamespace(
            n_layers=1,
            n_heads=1,
            n_queries=n_queries,
            deep_supervision=False,
            pos_encoding_type="sinusoidal",
            dim_feedforward=8,
            swapped_attention=False,
            masked_attention=False,
        ),
        point_encoder=SimpleNamespace(
            l3=SimpleNamespace(out_channels=feature_dim),
        ),
    )


def test_autodec_encoder_returns_superdec_outputs_features_and_residual():
    from autodec.encoder import AutoDecEncoder

    ctx = _ctx()
    encoder = AutoDecEncoder(
        ctx,
        point_encoder=FakePointEncoder(feature_dim=4),
        layers=FakeLayers(n_queries=2, feature_dim=4),
        heads=FakeHeads(),
    )

    out = encoder(torch.randn(2, 5, 3))

    assert out["assign_matrix"].shape == (2, 5, 2)
    assert out["point_features"].shape == (2, 5, 4)
    assert out["sq_features"].shape == (2, 2, 4)
    assert out["pooled_features"].shape == (2, 2, 4)
    assert out["residual"].shape == (2, 2, 3)
    assert out["exist_logit"].shape == (2, 2, 1)
    assert torch.allclose(out["assign_matrix"].sum(dim=-1), torch.ones(2, 5))
