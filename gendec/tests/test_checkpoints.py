import torch
import torch.nn as nn


def test_phase1_checkpoint_round_trip_restores_model_and_epoch(tmp_path):
    from gendec.training.checkpoints import load_phase1_checkpoint, save_phase1_checkpoint

    expected = nn.Linear(4, 2)
    model = nn.Linear(4, 2)
    path = tmp_path / "phase1.pt"

    save_phase1_checkpoint(expected, optimizer=None, scheduler=None, epoch=3, loss=1.25, path=path)
    meta = load_phase1_checkpoint(model, path)

    assert meta["epoch"] == 3
    assert meta["loss"] == 1.25
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected.state_dict()[key])
