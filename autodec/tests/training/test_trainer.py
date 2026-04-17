from types import SimpleNamespace

import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, points):
        decoded = points[:, :1, :] * 0.0 + self.bias
        return {
            "decoded_points": decoded,
            "decoded_weights": torch.ones(points.shape[0], 1, device=points.device),
        }


class TinyLoss(nn.Module):
    def forward(self, batch, outdict):
        loss = outdict["decoded_points"].square().mean()
        return loss, {"all": loss.detach().item()}


class RecordingVisualizer:
    def __init__(self):
        self.calls = []

    def write_epoch(self, batch, outdict, epoch, split, num_samples):
        self.calls.append(
            {
                "batch": batch,
                "outdict": outdict,
                "epoch": epoch,
                "split": split,
                "num_samples": num_samples,
            }
        )
        return ["record"]


class RecordingWandbRun:
    def __init__(self):
        self.logs = []

    def log(self, payload, step=None):
        self.logs.append((payload, step))


def test_autodec_trainer_runs_one_train_and_eval_epoch(tmp_path):
    from autodec.training.trainer import AutoDecTrainer

    dataloaders = {
        "train": [{"points": torch.ones(2, 3, 3)}],
        "val": [{"points": torch.ones(2, 3, 3)}],
    }
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = AutoDecTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataloaders=dataloaders,
        loss_fn=TinyLoss(),
        ctx=SimpleNamespace(num_epochs=1, save_path=str(tmp_path)),
        device=torch.device("cpu"),
    )

    train_metrics = trainer.train_one_epoch(0)
    val_metrics = trainer.evaluate(0)

    assert "all" in train_metrics
    assert "all" in val_metrics


def test_autodec_trainer_logs_visualizations_after_eval_epoch(tmp_path):
    from autodec.training.trainer import AutoDecTrainer

    dataloaders = {
        "train": [{"points": torch.ones(2, 3, 3)}],
        "val": [{"points": torch.ones(2, 3, 3)}],
    }
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    visualizer = RecordingVisualizer()
    wandb_run = RecordingWandbRun()

    trainer = AutoDecTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataloaders=dataloaders,
        loss_fn=TinyLoss(),
        ctx=SimpleNamespace(
            num_epochs=1,
            save_path=str(tmp_path),
            visualize_every_n_epochs=1,
            visualize_num_samples=2,
            visualize_split="val",
            log_visualizations_to_wandb=True,
        ),
        device=torch.device("cpu"),
        wandb_run=wandb_run,
        visualizer=visualizer,
        wandb_visual_log_builder=lambda records: {"visual/gt": records},
    )

    trainer.evaluate(4)

    assert len(visualizer.calls) == 1
    assert visualizer.calls[0]["epoch"] == 4
    assert visualizer.calls[0]["split"] == "val"
    assert visualizer.calls[0]["num_samples"] == 2
    assert wandb_run.logs[-1] == ({"visual/gt": ["record"]}, 4)
