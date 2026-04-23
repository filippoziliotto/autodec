from pathlib import Path

import torch
import yaml


def _write_autodec_decoder_assets(tmp_path):
    from autodec.decoder import AutoDecDecoder

    config = {
        "autodec": {
            "residual_dim": 4,
            "primitive_dim": 18,
            "n_surface_samples": 2,
            "exist_tau": 1.0,
            "decoder": {
                "hidden_dim": 16,
                "n_heads": 4,
                "positional_frequencies": 0,
                "component_feature_dim": 0,
                "n_blocks": 1,
                "self_attention_mode": "none",
                "offset_scale": None,
                "offset_cap": None,
                "detach_sq_for_recon": False,
            },
        }
    }
    config_path = tmp_path / "autodec_eval.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    decoder = AutoDecDecoder(
        residual_dim=4,
        primitive_dim=18,
        n_surface_samples=2,
        hidden_dim=16,
        n_heads=4,
        positional_frequencies=0,
        component_feature_dim=0,
        n_blocks=1,
        self_attention_mode="none",
        offset_cap=None,
        offset_scale=None,
    )
    for param in decoder.offset_decoder.parameters():
        torch.nn.init.zeros_(param)

    checkpoint_path = tmp_path / "autodec_decoder.pt"
    torch.save(
        {
            "model_state_dict": {
                f"decoder.{key}": value
                for key, value in decoder.state_dict().items()
            }
        },
        checkpoint_path,
    )
    return config_path, checkpoint_path


def test_autodec_bridge_decodes_sampled_scaffolds_with_zero_residual(tmp_path):
    from gendec.data.toy_builder import build_toy_example
    from gendec.eval.autodec_bridge import (
        build_frozen_autodec_decoder,
        decode_scaffolds_with_zero_residual,
    )
    from gendec.sampling import postprocess_tokens

    config_path, checkpoint_path = _write_autodec_decoder_assets(tmp_path)
    bridge = build_frozen_autodec_decoder(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    example = build_toy_example(model_id="chair_0001", num_points=32)
    processed = postprocess_tokens(
        example["tokens_e"].unsqueeze(0),
        stats={
            "mean": torch.zeros(15),
            "std": torch.ones(15),
        },
        exist_threshold=0.5,
    )

    decoded = decode_scaffolds_with_zero_residual(
        processed,
        decoder=bridge["decoder"],
        residual_dim=bridge["residual_dim"],
    )

    assert decoded["decoded_points"].shape == (1, 32, 3)
    assert decoded["surface_points"].shape == (1, 32, 3)
    assert torch.allclose(decoded["decoded_points"], decoded["surface_points"])
