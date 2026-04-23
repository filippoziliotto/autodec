import torch


def test_set_transformer_flow_model_matches_token_contract():
    from gendec.models.set_transformer_flow import SetTransformerFlowModel

    model = SetTransformerFlowModel(token_dim=15, hidden_dim=32, n_blocks=2, n_heads=4)

    et = torch.randn(3, 16, 15)
    t = torch.rand(3)
    out = model(et, t)

    assert out.shape == (3, 16, 15)
