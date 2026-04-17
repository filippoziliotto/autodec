import torch
import torch.nn as nn


def test_load_superdec_encoder_checkpoint_strips_module_prefix(tmp_path):
    from autodec.utils.checkpoints import load_superdec_encoder_checkpoint

    target = nn.Linear(2, 1)
    expected = nn.Linear(2, 1)
    checkpoint = {
        "model_state_dict": {
            f"module.{key}": value.clone()
            for key, value in expected.state_dict().items()
        }
    }
    path = tmp_path / "superdec.pt"
    torch.save(checkpoint, path)

    load_superdec_encoder_checkpoint(target, path)

    for key, value in target.state_dict().items():
        assert torch.equal(value, expected.state_dict()[key])


def test_load_autodec_checkpoint_restores_model_and_returns_epoch(tmp_path):
    from autodec.utils.checkpoints import load_autodec_checkpoint

    model = nn.Linear(2, 1)
    expected = nn.Linear(2, 1)
    path = tmp_path / "autodec.pt"
    torch.save(
        {
            "model_state_dict": expected.state_dict(),
            "epoch": 7,
            "val_loss": 0.25,
        },
        path,
    )

    meta = load_autodec_checkpoint(model, path)

    assert meta["epoch"] == 7
    assert meta["val_loss"] == 0.25
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected.state_dict()[key])
