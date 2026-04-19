import torch.nn as nn

from autodec.decoder import AutoDecDecoder
from autodec.encoder import AutoDecEncoder


class _EncoderCtx:
    """Proxy that adds AutoDec-level residual_dim to the encoder config."""

    def __init__(self, base, residual_dim):
        self._base = base
        self.residual_dim = residual_dim

    def __getattr__(self, name):
        return getattr(self._base, name)


def _cfg_get(cfg, name, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


class AutoDec(nn.Module):
    """Full AutoDec model: SuperDec-compatible encoder followed by decoder."""

    def __init__(self, ctx=None, encoder=None, decoder=None):
        super().__init__()
        if encoder is None:
            if ctx is None:
                raise ValueError("AutoDec requires ctx when encoder is not provided")
            encoder_ctx = _cfg_get(ctx, "encoder", ctx)
            residual_dim = _cfg_get(ctx, "residual_dim", _cfg_get(encoder_ctx, "residual_dim", 64))
            encoder = AutoDecEncoder(_EncoderCtx(encoder_ctx, residual_dim))
        if decoder is None:
            if ctx is None:
                raise ValueError("AutoDec requires ctx when decoder is not provided")
            decoder_ctx = _cfg_get(ctx, "decoder", None)
            residual_dim = _cfg_get(ctx, "residual_dim", 64)
            decoder = AutoDecDecoder(
                residual_dim=residual_dim,
                primitive_dim=_cfg_get(ctx, "primitive_dim", 18),
                n_surface_samples=_cfg_get(ctx, "n_surface_samples", 256),
                hidden_dim=_cfg_get(decoder_ctx, "hidden_dim", _cfg_get(ctx, "hidden_dim", 128)),
                n_heads=_cfg_get(decoder_ctx, "n_heads", _cfg_get(ctx, "n_heads", 4)),
                exist_tau=_cfg_get(ctx, "exist_tau", 1.0),
                offset_scale=_cfg_get(decoder_ctx, "offset_scale", _cfg_get(ctx, "offset_scale", None)),
                positional_frequencies=_cfg_get(decoder_ctx, "positional_frequencies", 6),
                component_feature_dim=_cfg_get(decoder_ctx, "component_feature_dim", None),
                n_blocks=_cfg_get(decoder_ctx, "n_blocks", 2),
                self_attention_mode=_cfg_get(decoder_ctx, "self_attention_mode", "within_primitive"),
            )
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, points, return_attention=False, return_consistency=False):
        outdict = self.encoder(points)
        return self.decoder(
            outdict,
            return_attention=return_attention,
            return_consistency=return_consistency,
        )

    def _set_requires_grad(self, module, value):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = value

    def freeze_encoder_backbone(self):
        """Freeze pretrained SuperDec parts while leaving residuals trainable."""

        self._set_requires_grad(getattr(self.encoder, "point_encoder", None), False)
        self._set_requires_grad(getattr(self.encoder, "layers", None), False)
        self._set_requires_grad(getattr(self.encoder, "heads", None), False)
        self._set_requires_grad(getattr(self.encoder, "residual_projector", None), True)
        self._set_requires_grad(self.decoder, True)

    def unfreeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = True

    def phase1_parameters(self):
        yield from self.residual_parameters()
        yield from self.decoder_parameters()

    def encoder_backbone_parameters(self):
        for name in ("point_encoder", "layers", "heads"):
            module = getattr(self.encoder, name, None)
            if module is not None:
                yield from module.parameters()

    def residual_parameters(self):
        module = getattr(self.encoder, "residual_projector", None)
        if module is not None:
            yield from module.parameters()

    def decoder_parameters(self):
        yield from self.decoder.parameters()
