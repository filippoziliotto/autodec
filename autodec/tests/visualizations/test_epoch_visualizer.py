import json

import torch


def _batch():
    return {
        "points": torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ]
        )
    }


def _outdict():
    return {
        "scale": torch.ones(1, 1, 3),
        "shape": torch.ones(1, 1, 2),
        "rotate": torch.eye(3).view(1, 1, 3, 3),
        "trans": torch.zeros(1, 1, 3),
        "exist": torch.ones(1, 1, 1),
        "decoded_points": torch.tensor(
            [
                [
                    [0.0, 0.0, 0.1],
                    [1.0, 0.0, 0.1],
                    [0.0, 1.0, 0.1],
                ]
            ]
        ),
    }


def test_epoch_visualizer_writes_gt_sq_mesh_and_reconstruction(tmp_path):
    from autodec.visualizations import AutoDecEpochVisualizer

    visualizer = AutoDecEpochVisualizer(
        root_dir=tmp_path,
        run_name="debug_run",
        mesh_resolution=6,
        max_points=None,
    )

    records = visualizer.write_epoch(
        batch=_batch(),
        outdict=_outdict(),
        epoch=3,
        split="train",
        num_samples=1,
    )

    assert len(records) == 1
    record = records[0]
    assert record.input_path.exists()
    assert record.sq_mesh_path.exists()
    assert record.reconstruction_path.exists()
    assert record.metadata_path.exists()
    assert record.input_path.name == "input_gt.ply"
    assert record.reconstruction_path.name == "reconstruction.ply"
    assert record.sq_mesh_path.name == "sq_mesh.ply"

    metadata = json.loads(record.metadata_path.read_text())
    assert metadata["epoch"] == 3
    assert metadata["split"] == "train"
    assert metadata["sample_index"] == 0
    assert metadata["input_points"] == 3
    assert metadata["reconstruction_points"] == 3
    assert metadata["active_primitives"] == 1


def test_build_wandb_log_returns_expected_visual_keys(tmp_path):
    from autodec.visualizations import AutoDecEpochVisualizer, build_wandb_log

    visualizer = AutoDecEpochVisualizer(
        root_dir=tmp_path,
        run_name="debug_run",
        mesh_resolution=6,
        max_points=None,
    )
    records = visualizer.write_epoch(
        batch=_batch(),
        outdict=_outdict(),
        epoch=0,
        split="val",
        num_samples=1,
    )

    seen = []

    def fake_object3d(path):
        seen.append(path.name)
        return f"object:{path.name}"

    payload = build_wandb_log(records, object3d_factory=fake_object3d)

    assert payload == {
        "visual/gt": ["object:input_gt.ply"],
        "visual/sq_mesh": ["object:sq_mesh.ply"],
        "visual/reconstruction": ["object:reconstruction.ply"],
    }
    assert seen == ["input_gt.ply", "sq_mesh.ply", "reconstruction.ply"]


def test_visualizations_folder_has_same_name_documentation():
    doc = "autodec/visualizations/visualizations.md"

    with open(doc) as handle:
        text = handle.read()

    assert "AutoDecEpochVisualizer" in text
    assert "data/viz" in text
    assert "WandB" in text
