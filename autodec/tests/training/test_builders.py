from types import SimpleNamespace
import sys

import torch
import torch.nn as nn
from torch.utils.data import Subset


class TinyAutoDec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimpleNamespace(
            point_encoder=nn.Linear(2, 2),
            layers=nn.Linear(2, 2),
            heads=nn.Linear(2, 2),
            residual_projector=nn.Linear(2, 2),
        )
        self.decoder = nn.Linear(2, 2)

    def parameters(self):
        for module in (
            self.encoder.point_encoder,
            self.encoder.layers,
            self.encoder.heads,
            self.encoder.residual_projector,
            self.decoder,
        ):
            yield from module.parameters()

    def freeze_encoder_backbone(self):
        for module in (self.encoder.point_encoder, self.encoder.layers, self.encoder.heads):
            for param in module.parameters():
                param.requires_grad = False
        for param in self.encoder.residual_projector.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = True

    def phase1_parameters(self):
        return list(self.encoder.residual_projector.parameters()) + list(self.decoder.parameters())

    def encoder_backbone_parameters(self):
        params = []
        for module in (self.encoder.point_encoder, self.encoder.layers, self.encoder.heads):
            params.extend(module.parameters())
        return params

    def residual_parameters(self):
        return self.encoder.residual_projector.parameters()

    def decoder_parameters(self):
        return self.decoder.parameters()


def _cfg(phase=1):
    return SimpleNamespace(
        loss=SimpleNamespace(phase=phase),
        optimizer=SimpleNamespace(
            lr=1e-3,
            decoder_lr=2e-3,
            encoder_lr=1e-4,
            residual_lr=3e-3,
            weight_decay=0.0,
            betas=(0.9, 0.999),
        ),
    )


def test_build_loss_constructs_autodec_loss_from_cfg():
    from autodec.losses import AutoDecLoss
    from autodec.training.builders import build_loss

    cfg = SimpleNamespace(
        loss=SimpleNamespace(
            phase=2,
            lambda_sq=2.0,
            lambda_par=3.0,
            lambda_exist=4.0,
            lambda_cons=0.0,
            n_sq_samples=8,
        )
    )

    loss = build_loss(cfg)

    assert isinstance(loss, AutoDecLoss)
    assert loss.phase == 2
    assert loss.lambda_sq == 2.0


def test_build_optimizer_phase1_trains_residual_and_decoder_only():
    from autodec.training.builders import build_optimizer

    model = TinyAutoDec()
    optimizer = build_optimizer(_cfg(phase=1), model)
    optimized = {id(param) for group in optimizer.param_groups for param in group["params"]}

    assert not any(p.requires_grad for p in model.encoder.point_encoder.parameters())
    assert optimized == {id(param) for param in model.phase1_parameters()}


def test_build_optimizer_phase2_uses_differential_learning_rates():
    from autodec.training.builders import build_optimizer

    model = TinyAutoDec()
    optimizer = build_optimizer(_cfg(phase=2), model)
    lrs = [group["lr"] for group in optimizer.param_groups]

    assert lrs == [1e-4, 3e-3, 2e-3]
    assert all(p.requires_grad for p in model.parameters())


def test_build_visualizer_uses_visualization_config(tmp_path):
    from autodec.training.builders import build_visualizer
    from autodec.visualizations import AutoDecEpochVisualizer

    cfg = SimpleNamespace(
        run_name="run_a",
        visualization=SimpleNamespace(
            enabled=True,
            root_dir=str(tmp_path),
            mesh_resolution=12,
            exist_threshold=0.25,
            max_points=32,
        ),
    )

    visualizer = build_visualizer(cfg)

    assert isinstance(visualizer, AutoDecEpochVisualizer)
    assert visualizer.root_dir == tmp_path
    assert visualizer.run_name == "run_a"
    assert visualizer.mesh_resolution == 12
    assert visualizer.exist_threshold == 0.25
    assert visualizer.max_points == 32


def test_build_wandb_run_is_lazy_uses_project_and_env_key(monkeypatch):
    from autodec.training.builders import build_wandb_run

    calls = []

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            calls.append(kwargs)
            return "run"

    monkeypatch.setattr("autodec.training.builders._import_wandb", lambda: FakeWandb)
    monkeypatch.setenv("AUTODEC_WANDB_KEY", "secret")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = SimpleNamespace(
        use_wandb=True,
        run_name="autodec_debug",
        wandb=SimpleNamespace(
            project="autodec",
            entity="ignored-user",
            api_key_env="AUTODEC_WANDB_KEY",
        ),
    )

    run = build_wandb_run(cfg)

    assert run == "run"
    assert calls[0].get("entity") is None
    assert calls == [
        {
            "project": "autodec",
            "name": "autodec_debug",
        }
    ]
    import os

    assert os.environ["WANDB_API_KEY"] == "secret"


def test_build_wandb_run_returns_none_when_disabled(monkeypatch):
    from autodec.training.builders import build_wandb_run

    monkeypatch.setattr(
        "autodec.training.builders._import_wandb",
        lambda: (_ for _ in ()).throw(AssertionError("wandb should not import")),
    )

    assert build_wandb_run(SimpleNamespace(use_wandb=False)) is None


def test_limit_dataset_returns_deterministic_subset_when_limit_is_set():
    from autodec.training.builders import limit_dataset

    dataset = list(range(10))

    subset_a = limit_dataset(dataset, max_items=4, seed=7)
    subset_b = limit_dataset(dataset, max_items=4, seed=7)
    subset_c = limit_dataset(dataset, max_items=4, seed=8)

    assert isinstance(subset_a, Subset)
    assert len(subset_a) == 4
    assert subset_a.indices == subset_b.indices
    assert subset_a.indices != subset_c.indices
    assert sorted(subset_a.indices) != subset_a.indices


def test_limit_dataset_leaves_dataset_unchanged_when_limit_is_null_or_large():
    from autodec.training.builders import limit_dataset

    dataset = list(range(3))

    assert limit_dataset(dataset, max_items=None, seed=7) is dataset
    assert limit_dataset(dataset, max_items=10, seed=7) is dataset


def test_build_dataloaders_applies_shapenet_category_split(monkeypatch):
    from autodec.utils.shapenet_categories import PAPER_SEEN_CATEGORIES
    from autodec.training.builders import build_dataloaders

    calls = []

    class FakeShapeNet:
        def __init__(self, split, cfg):
            calls.append((split, list(cfg.shapenet.categories)))

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"points": torch.zeros(4, 3)}

    fake_module = SimpleNamespace(ABO=object, ASE_Object=object, ShapeNet=FakeShapeNet)
    monkeypatch.setitem(sys.modules, "superdec.data.dataloader", fake_module)
    cfg = SimpleNamespace(
        dataset="shapenet",
        shapenet=SimpleNamespace(
            category_split="paper_seen",
            categories=None,
            max_train_items=None,
            max_val_items=None,
            subset_seed=0,
        ),
        trainer=SimpleNamespace(batch_size=1, num_workers=0),
    )

    build_dataloaders(cfg)

    assert calls == [
        ("train", PAPER_SEEN_CATEGORIES),
        ("val", PAPER_SEEN_CATEGORIES),
    ]
