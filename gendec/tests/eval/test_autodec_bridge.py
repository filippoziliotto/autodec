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


def test_phase2_autodec_bridge_hard_thresholds_existence_inputs():
    from gendec.eval.autodec_bridge import decode_joint_scaffolds

    class RecordingDecoder:
        def __init__(self):
            self.last_outdict = None

        def __call__(self, outdict, return_attention=False, return_consistency=False):
            self.last_outdict = outdict
            batch, primitives = outdict["scale"].shape[:2]
            samples = 2
            return {
                "decoded_points": outdict["scale"].new_zeros(batch, primitives * samples, 3),
                "surface_points": outdict["scale"].new_zeros(batch, primitives * samples, 3),
                "decoded_weights": outdict["scale"].new_zeros(batch, primitives * samples),
                "part_ids": torch.arange(primitives, device=outdict["scale"].device).repeat_interleave(samples),
            }

    processed = {
        "scale": torch.ones(1, 3, 3),
        "shape": torch.ones(1, 3, 2),
        "rotate": torch.eye(3).view(1, 1, 3, 3).repeat(1, 3, 1, 1),
        "trans": torch.zeros(1, 3, 3),
        "exist_logit": torch.tensor([[[0.2], [-0.3], [0.7]]], dtype=torch.float32),
        "exist": torch.tensor([[[0.55], [0.42], [0.91]]], dtype=torch.float32),
        "active_mask": torch.tensor([[True, False, True]]),
        "tokens_z": torch.randn(1, 3, 4),
    }
    decoder = RecordingDecoder()

    decode_joint_scaffolds(processed, decoder=decoder)

    assert decoder.last_outdict is not None
    assert torch.equal(
        decoder.last_outdict["exist"],
        torch.tensor([[[1.0], [0.0], [1.0]]]),
    )
    assert torch.equal(
        decoder.last_outdict["exist_logit"],
        torch.tensor([[[20.0], [-20.0], [20.0]]]),
    )
    assert torch.equal(decoder.last_outdict["residual"], processed["tokens_z"])


def test_prune_points_by_active_primitives_filters_inactive_parts():
    from gendec.utils.inference import prune_points_by_active_primitives

    outdict = {
        "decoded_points": torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                ]
            ]
        ),
        "part_ids": torch.tensor([0, 0, 1, 1, 2, 2]),
        "exist": torch.tensor([[[1.0], [0.0], [1.0]]]),
    }

    pruned = prune_points_by_active_primitives(
        outdict,
        "decoded_points",
        exist_threshold=0.5,
        target_count=None,
    )

    assert len(pruned) == 1
    assert torch.equal(
        pruned[0],
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        ),
    )
