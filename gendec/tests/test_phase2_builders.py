import tempfile


def _make_phase2_cfg(root, checkpoint_path):
    return {
        "dataset": {"root": root, "split": "train", "val_split": "val"},
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "num_epochs": 1,
            "checkpoint_path": checkpoint_path,
            "disable_tqdm": True,
        },
        "model": {
            "explicit_dim": 15,
            "residual_dim": 64,
            "hidden_dim": 32,
            "n_blocks": 2,
            "n_heads": 4,
        },
        "loss": {
            "explicit_dim": 15,
            "lambda_e": 1.0,
            "lambda_z": 1.0,
            "lambda_exist": 0.05,
        },
        "optimizer": {"name": "AdamW", "lr": 1e-4},
    }


def test_build_phase2_dataset_returns_joint_token_dataset():
    from gendec.data.toy_builder import write_toy_phase2_dataset
    from gendec.training.builders import build_phase2_dataset

    with tempfile.TemporaryDirectory() as tmp:
        write_toy_phase2_dataset(tmp, split="train", num_examples=4, num_points=256)
        cfg = _make_phase2_cfg(tmp, f"{tmp}/ckpt.pt")
        ds = build_phase2_dataset(cfg)
        assert len(ds) > 0
        item = ds[0]
        assert "tokens_ez" in item


def test_build_phase2_dataloader_returns_dataset_and_loader():
    from gendec.data.toy_builder import write_toy_phase2_dataset
    from gendec.training.builders import build_phase2_dataloader

    with tempfile.TemporaryDirectory() as tmp:
        write_toy_phase2_dataset(tmp, split="train", num_examples=4, num_points=256)
        cfg = _make_phase2_cfg(tmp, f"{tmp}/ckpt.pt")
        ds, loader = build_phase2_dataloader(cfg, batch_size=2)
        batch = next(iter(loader))
        assert batch["tokens_ez"].shape[0] == 2


def test_build_phase2_model_returns_joint_model():
    from gendec.models.set_transformer_flow import JointSetTransformerFlowModel
    from gendec.training.builders import build_phase2_model

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_phase2_cfg(tmp, f"{tmp}/ckpt.pt")
        model = build_phase2_model(cfg)
        assert isinstance(model, JointSetTransformerFlowModel)


def test_build_phase2_loss_returns_joint_loss():
    from gendec.losses.flow_matching import JointFlowMatchingLoss
    from gendec.training.builders import build_phase2_loss

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_phase2_cfg(tmp, f"{tmp}/ckpt.pt")
        loss_fn = build_phase2_loss(cfg)
        assert isinstance(loss_fn, JointFlowMatchingLoss)


def test_build_phase2_train_val_dataloaders_returns_loaders():
    from gendec.data.toy_builder import write_toy_phase2_dataset
    from gendec.training.builders import build_phase2_train_val_dataloaders

    with tempfile.TemporaryDirectory() as tmp:
        write_toy_phase2_dataset(tmp, split="train", num_examples=4, num_points=256)
        cfg = _make_phase2_cfg(tmp, f"{tmp}/ckpt.pt")
        datasets, loaders = build_phase2_train_val_dataloaders(cfg)
        assert loaders["train"] is not None
        # val may be None (missing split) or a loader (fallback to all examples)
        assert "val" in loaders
