from types import SimpleNamespace

import torch
import torch.nn as nn


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = nn.Linear(3, 4)
        self.layers = nn.Linear(4, 4)
        self.heads = nn.Linear(4, 2)
        self.residual_projector = nn.Linear(2, 2)

    def forward(self, points):
        batch = points.shape[0]
        return {
            "scale": torch.ones(batch, 1, 3, device=points.device),
            "shape": torch.ones(batch, 1, 2, device=points.device),
            "rotate": torch.eye(3, device=points.device).view(1, 1, 3, 3).repeat(batch, 1, 1, 1),
            "trans": torch.zeros(batch, 1, 3, device=points.device),
            "exist_logit": torch.zeros(batch, 1, 1, device=points.device),
            "exist": torch.ones(batch, 1, 1, device=points.device) * 0.5,
            "assign_matrix": torch.ones(batch, points.shape[1], 1, device=points.device),
            "residual": torch.zeros(batch, 1, 2, device=points.device),
        }


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.return_consistency_flags = []

    def forward(self, outdict, return_attention=False, return_consistency=False):
        self.return_consistency_flags.append(return_consistency)
        batch = outdict["scale"].shape[0]
        decoded = torch.zeros(batch, 2, 3, device=outdict["scale"].device) + self.weight
        result = dict(outdict)
        result["decoded_points"] = decoded
        result["decoded_weights"] = torch.ones(batch, 2, device=decoded.device)
        if return_attention:
            result["decoder_attention"] = torch.ones(batch, 2, 1, device=decoded.device)
        if return_consistency:
            result["consistency_decoded_points"] = decoded + 1.0
        return result


def test_autodec_wrapper_runs_encoder_then_decoder_and_exports_attention():
    from autodec.autodec import AutoDec

    model = AutoDec(encoder=ToyEncoder(), decoder=ToyDecoder())

    out = model(torch.randn(3, 5, 3), return_attention=True)

    assert out["decoded_points"].shape == (3, 2, 3)
    assert out["assign_matrix"].shape == (3, 5, 1)
    assert out["decoder_attention"].shape == (3, 2, 1)


def test_autodec_wrapper_forwards_consistency_request():
    from autodec.autodec import AutoDec

    decoder = ToyDecoder()
    model = AutoDec(encoder=ToyEncoder(), decoder=decoder)

    out = model(torch.randn(3, 5, 3), return_consistency=True)

    assert decoder.return_consistency_flags == [True]
    assert out["consistency_decoded_points"].shape == (3, 2, 3)


def test_autodec_phase1_freezes_superdec_backbone_but_not_residual_or_decoder():
    from autodec.autodec import AutoDec

    model = AutoDec(encoder=ToyEncoder(), decoder=ToyDecoder())

    model.freeze_encoder_backbone()

    assert not any(p.requires_grad for p in model.encoder.point_encoder.parameters())
    assert not any(p.requires_grad for p in model.encoder.layers.parameters())
    assert not any(p.requires_grad for p in model.encoder.heads.parameters())
    assert all(p.requires_grad for p in model.encoder.residual_projector.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())
    assert {id(p) for p in model.phase1_parameters()} == {
        id(p)
        for module in (model.encoder.residual_projector, model.decoder)
        for p in module.parameters()
    }

    model.unfreeze_encoder()

    assert all(p.requires_grad for p in model.parameters())


def test_autodec_wrapper_passes_decoder_detach_sq_for_recon_config():
    from autodec.autodec import AutoDec

    ctx = SimpleNamespace(
        residual_dim=4,
        primitive_dim=18,
        n_surface_samples=2,
        exist_tau=1.0,
        decoder=SimpleNamespace(
            hidden_dim=16,
            n_heads=4,
            positional_frequencies=0,
            component_feature_dim=0,
            n_blocks=1,
            self_attention_mode="none",
            offset_scale=None,
            offset_cap=None,
            detach_sq_for_recon=True,
        ),
    )

    model = AutoDec(ctx=ctx, encoder=ToyEncoder())

    assert model.decoder.detach_sq_for_recon is True
