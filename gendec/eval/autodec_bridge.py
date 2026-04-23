from pathlib import Path

import numpy as np
import torch

from gendec.config import cfg_get, load_yaml_config


def _lazy_autodec_decoder():
    from autodec.decoder import AutoDecDecoder
    from autodec.utils.checkpoints import extract_state_dict, strip_module_prefix

    return AutoDecDecoder, extract_state_dict, strip_module_prefix


class _FallbackAngleSampler:
    def __init__(self, n_samples):
        self.n_samples = int(n_samples)

    def sample_on_batch(self, scale, shape):
        batch, primitives = scale.shape[:2]
        etas = np.linspace(
            -np.pi / 2 + 1e-3,
            np.pi / 2 - 1e-3,
            self.n_samples,
            dtype=np.float32,
        )
        omegas = np.linspace(
            -np.pi + 1e-3,
            np.pi - 1e-3,
            self.n_samples,
            dtype=np.float32,
        )
        return (
            np.broadcast_to(etas, (batch, primitives, self.n_samples)),
            np.broadcast_to(omegas, (batch, primitives, self.n_samples)),
        )


def _supports_equal_distance_sampler():
    try:
        import superdec.fast_sampler  # noqa: F401

        return True
    except Exception:
        return False


def _decoder_kwargs(autodec_cfg):
    decoder_cfg = cfg_get(autodec_cfg, "decoder")
    residual_dim = int(cfg_get(autodec_cfg, "residual_dim", 64))
    return {
        "residual_dim": residual_dim,
        "primitive_dim": int(cfg_get(autodec_cfg, "primitive_dim", 18)),
        "n_surface_samples": int(cfg_get(autodec_cfg, "n_surface_samples", 256)),
        "hidden_dim": int(cfg_get(decoder_cfg, "hidden_dim", cfg_get(autodec_cfg, "hidden_dim", 128))),
        "n_heads": int(cfg_get(decoder_cfg, "n_heads", cfg_get(autodec_cfg, "n_heads", 4))),
        "exist_tau": float(cfg_get(autodec_cfg, "exist_tau", 1.0)),
        "offset_scale": cfg_get(decoder_cfg, "offset_scale", cfg_get(autodec_cfg, "offset_scale", None)),
        "offset_cap": cfg_get(decoder_cfg, "offset_cap", cfg_get(autodec_cfg, "offset_cap", None)),
        "positional_frequencies": int(cfg_get(decoder_cfg, "positional_frequencies", 6)),
        "component_feature_dim": cfg_get(decoder_cfg, "component_feature_dim", None),
        "n_blocks": int(cfg_get(decoder_cfg, "n_blocks", 2)),
        "self_attention_mode": cfg_get(decoder_cfg, "self_attention_mode", "within_primitive"),
        "detach_sq_for_recon": bool(
            cfg_get(
                decoder_cfg,
                "detach_sq_for_recon",
                cfg_get(autodec_cfg, "detach_sq_for_recon", False),
            )
        ),
    }


def build_frozen_autodec_decoder(config_path, checkpoint_path=None, device="cpu"):
    AutoDecDecoder, extract_state_dict, strip_module_prefix = _lazy_autodec_decoder()

    config = load_yaml_config(config_path)
    autodec_cfg = cfg_get(config, "autodec", config)
    if autodec_cfg is None:
        raise ValueError("AutoDec decoder config must contain an 'autodec' section or be that section itself.")

    if checkpoint_path is None:
        checkpoint_path = cfg_get(cfg_get(config, "checkpoints"), "resume_from")
    if checkpoint_path is None:
        raise ValueError("AutoDec decoder loading requires a checkpoint path.")

    decoder = AutoDecDecoder(**_decoder_kwargs(autodec_cfg)).to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    decoder_state = {
        key.removeprefix("decoder."): value
        for key, value in state_dict.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        raise ValueError(f"No decoder weights found in checkpoint: {checkpoint_path}")
    decoder.load_state_dict(decoder_state)
    if not _supports_equal_distance_sampler():
        decoder.surface_sampler.angle_sampler = _FallbackAngleSampler(decoder.n_surface_samples)
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    return {
        "decoder": decoder,
        "residual_dim": int(cfg_get(autodec_cfg, "residual_dim", 64)),
        "config_path": Path(config_path),
        "checkpoint_path": Path(checkpoint_path),
    }


def sampled_scaffolds_to_decoder_outdict(processed, residual_dim):
    """Build a decoder input outdict using zero residuals (Phase 1 bridge)."""
    batch, n_prim = processed["scale"].shape[:2]
    residual = processed["scale"].new_zeros(batch, n_prim, int(residual_dim))
    return {
        "scale": processed["scale"],
        "shape": processed["shape"],
        "rotate": processed["rotate"],
        "trans": processed["trans"],
        "exist_logit": processed["exist_logit"],
        "exist": processed["exist"],
        "residual": residual,
    }


def sampled_joint_scaffolds_to_decoder_outdict(processed):
    """Build a decoder input outdict using generated residuals (Phase 2 bridge).

    ``processed`` is the output of ``postprocess_joint_tokens`` / ``sample_joint_scaffolds``
    and must contain ``tokens_z`` with shape [B, 16, residual_dim].
    """
    return {
        "scale": processed["scale"],
        "shape": processed["shape"],
        "rotate": processed["rotate"],
        "trans": processed["trans"],
        "exist_logit": processed["exist_logit"],
        "exist": processed["exist"],
        "residual": processed["tokens_z"],
    }


def decode_scaffolds_with_zero_residual(processed, decoder, residual_dim, return_attention=False):
    """Phase 1 decode: pass scaffold through frozen AutoDec decoder with Z=0."""
    outdict = sampled_scaffolds_to_decoder_outdict(processed, residual_dim=residual_dim)
    with torch.no_grad():
        return decoder(
            outdict,
            return_attention=return_attention,
            return_consistency=False,
        )


def decode_joint_scaffolds(processed, decoder, return_attention=False):
    """Phase 2 decode: pass joint (E, Z) through frozen AutoDec decoder.

    Uses the generated residual tokens ``processed["tokens_z"]`` instead of
    zeroing the residual branch.

    Args:
        processed:       output of ``sample_joint_scaffolds`` / ``postprocess_joint_tokens``
        decoder:         frozen AutoDecDecoder (from ``build_frozen_autodec_decoder``)
        return_attention: whether to return decoder attention weights

    Returns:
        outdict from the AutoDec decoder, containing at minimum
        ``decoded_points``, ``decoded_offsets``, ``decoded_weights``,
        and ``surface_points``.
    """
    outdict = sampled_joint_scaffolds_to_decoder_outdict(processed)
    with torch.no_grad():
        return decoder(
            outdict,
            return_attention=return_attention,
            return_consistency=False,
        )
