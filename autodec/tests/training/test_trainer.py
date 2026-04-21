import json
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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


class ConsistencyLoss(nn.Module):
    lambda_cons = 1.0

    def forward(self, batch, outdict):
        assert "consistency_decoded_points" in outdict
        loss = outdict["consistency_decoded_points"].square().mean()
        return loss, {"all": loss.detach().item(), "consistency_loss": loss.detach().item()}


class ConsistencyTinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))
        self.return_consistency_flags = []

    def forward(self, points, return_consistency=False):
        self.return_consistency_flags.append(return_consistency)
        decoded = points[:, :1, :] * 0.0 + self.bias
        outdict = {
            "decoded_points": decoded,
            "decoded_weights": torch.ones(points.shape[0], 1, device=points.device),
        }
        if return_consistency:
            outdict["consistency_decoded_points"] = decoded + 1.0
        return outdict


class PrunableVisualizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, points):
        batch_size = points.shape[0]
        active = torch.zeros(batch_size, 2, 3, device=points.device) + self.bias
        inactive = torch.ones(batch_size, 2, 3, device=points.device) * 100.0
        return {
            "decoded_points": torch.cat([active, inactive], dim=1),
            "decoded_weights": torch.ones(batch_size, 4, device=points.device),
            "part_ids": torch.tensor([0, 0, 1, 1], device=points.device),
            "exist": torch.tensor([[[0.9], [0.1]]], device=points.device).repeat(batch_size, 1, 1),
        }


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


class CategoryDataset(Dataset):
    def __init__(self):
        self.models = [
            {"category": "a", "model_id": "a0"},
            {"category": "a", "model_id": "a1"},
            {"category": "b", "model_id": "b0"},
            {"category": "c", "model_id": "c0"},
        ]

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        return {
            "points": torch.ones(3, 3) * float(index),
            "idx": index,
            "model_id": self.models[index]["model_id"],
        }


class RecordingWandbRun:
    def __init__(self):
        self.logs = []

    def log(self, payload, step=None):
        self.logs.append((payload, step))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


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


def test_autodec_trainer_writes_epoch_metrics_after_train_and_eval(tmp_path):
    from autodec.training.metric_logger import EpochMetricLogger
    from autodec.training.trainer import AutoDecTrainer

    dataloaders = {
        "train": [{"points": torch.ones(2, 3, 3)}],
        "val": [{"points": torch.ones(2, 3, 3)}],
    }
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    metrics_path = tmp_path / "metrics.jsonl"
    metric_logger = EpochMetricLogger(metrics_path, append=False)
    trainer = AutoDecTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataloaders=dataloaders,
        loss_fn=TinyLoss(),
        ctx=SimpleNamespace(
            num_epochs=1,
            save_path=str(tmp_path),
            save_every_n_epochs=10,
            evaluate_every_n_epochs=1,
        ),
        device=torch.device("cpu"),
        metric_logger=metric_logger,
    )

    trainer.train()

    assert metrics_path.exists()
    rows = _read_jsonl(metrics_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["epoch"] == 1
    assert row["epoch_index"] == 0
    assert "all" in row["train"]
    assert "all" in row["val"]
    assert row["evaluated"] is True
    assert row["val_loss"] == row["val"]["all"]
    assert row["lr"] == [0.01]


def test_autodec_trainer_logs_train_metrics_when_epoch_is_not_evaluated(tmp_path):
    from autodec.training.metric_logger import EpochMetricLogger
    from autodec.training.trainer import AutoDecTrainer

    dataloaders = {
        "train": [{"points": torch.ones(2, 3, 3)}],
        "val": [{"points": torch.ones(2, 3, 3)}],
    }
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    metrics_path = tmp_path / "metrics.jsonl"
    metric_logger = EpochMetricLogger(metrics_path, append=False)
    trainer = AutoDecTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataloaders=dataloaders,
        loss_fn=TinyLoss(),
        ctx=SimpleNamespace(
            num_epochs=2,
            save_path=str(tmp_path),
            save_every_n_epochs=10,
            evaluate_every_n_epochs=2,
        ),
        device=torch.device("cpu"),
        metric_logger=metric_logger,
    )

    trainer.train()

    rows = _read_jsonl(metrics_path)
    assert len(rows) == 2
    first = rows[0]
    assert "all" in first["train"]
    assert first["val"] is None
    assert first["evaluated"] is False
    assert first["val_loss"] is None


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


def test_autodec_trainer_prunes_reconstruction_points_for_visualization(tmp_path):
    from autodec.training.trainer import AutoDecTrainer

    dataloaders = {
        "train": [{"points": torch.zeros(1, 4, 3)}],
        "val": [{"points": torch.zeros(1, 4, 3)}],
    }
    model = PrunableVisualizationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    visualizer = RecordingVisualizer()

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
            visualize_num_samples=1,
            visualize_split="val",
            log_visualizations_to_wandb=False,
        ),
        device=torch.device("cpu"),
        visualizer=visualizer,
    )

    trainer.evaluate(0)

    decoded_points = visualizer.calls[0]["outdict"]["decoded_points"]
    assert decoded_points.shape == (1, 4, 3)
    assert decoded_points.max().item() == 0.0


def test_autodec_trainer_logs_one_visualization_per_category_by_default(tmp_path):
    from autodec.training.trainer import AutoDecTrainer

    dataset = CategoryDataset()
    dataloaders = {
        "train": [{"points": torch.ones(2, 3, 3)}],
        "val": DataLoader(dataset, batch_size=2, shuffle=False),
    }
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    visualizer = RecordingVisualizer()

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
            visualize_num_samples=1,
            visualize_samples_per_category=1,
            visualize_category_balanced=True,
            visualize_split="val",
            log_visualizations_to_wandb=False,
        ),
        device=torch.device("cpu"),
        visualizer=visualizer,
    )

    trainer.evaluate(0)

    call = visualizer.calls[0]
    assert call["num_samples"] == 3
    assert call["batch"]["idx"].tolist() == [0, 2, 3]
    assert call["batch"]["points"].shape[0] == 3


def test_autodec_trainer_requests_consistency_pass_only_when_loss_needs_it(tmp_path):
    from autodec.training.trainer import AutoDecTrainer

    dataloaders = {
        "train": [{"points": torch.ones(2, 3, 3)}],
        "val": [{"points": torch.ones(2, 3, 3)}],
    }
    model = ConsistencyTinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = AutoDecTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataloaders=dataloaders,
        loss_fn=ConsistencyLoss(),
        ctx=SimpleNamespace(num_epochs=1, save_path=str(tmp_path)),
        device=torch.device("cpu"),
    )

    trainer.train_one_epoch(0)

    assert model.return_consistency_flags == [True]
