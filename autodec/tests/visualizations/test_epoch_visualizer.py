import json
import sys
from types import SimpleNamespace

import numpy as np
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


def _two_primitive_outdict():
    outdict = _outdict()
    outdict["scale"] = torch.ones(1, 2, 3)
    outdict["shape"] = torch.ones(1, 2, 2)
    outdict["rotate"] = torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    outdict["trans"] = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    outdict["exist"] = torch.ones(1, 2, 1)
    return outdict


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
    assert record.sq_mesh_path.name == "sq_mesh.obj"

    metadata = json.loads(record.metadata_path.read_text())
    assert metadata["epoch"] == 3
    assert metadata["split"] == "train"
    assert metadata["sample_index"] == 0
    assert metadata["input_points"] == 3
    assert metadata["reconstruction_points"] == 3
    assert metadata["active_primitives"] == 1


def test_epoch_visualizer_writes_optional_lm_sq_mesh(tmp_path):
    from autodec.visualizations import AutoDecEpochVisualizer

    visualizer = AutoDecEpochVisualizer(
        root_dir=tmp_path,
        run_name="debug_run",
        mesh_resolution=6,
        max_points=None,
    )
    lm_outdict = _outdict()
    lm_outdict["trans"] = torch.tensor([[[1.0, 0.0, 0.0]]])

    records = visualizer.write_epoch(
        batch=_batch(),
        outdict=_outdict(),
        lm_outdict=lm_outdict,
        epoch=3,
        split="test",
        num_samples=1,
    )

    record = records[0]
    assert record.sq_mesh_path.exists()
    assert record.sq_mesh_lm_path.exists()
    assert record.sq_mesh_lm_path.name == "sq_mesh_lm.obj"


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
        "visual/sq_mesh": ["object:sq_mesh.obj"],
        "visual/reconstruction": ["object:reconstruction.ply"],
    }
    assert seen == ["input_gt.ply", "sq_mesh.obj", "reconstruction.ply"]


def test_default_wandb_log_converts_point_cloud_ply_to_arrays(tmp_path, monkeypatch):
    from autodec.visualizations import AutoDecEpochVisualizer, build_wandb_log

    captured = []

    class FakeObject3D:
        def __init__(self, value):
            captured.append(value)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Object3D=FakeObject3D))

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

    payload = build_wandb_log(records)

    assert set(payload) == {"visual/gt", "visual/sq_mesh", "visual/reconstruction"}
    assert captured[0].shape == (3, 6)
    assert captured[1].endswith("sq_mesh.obj")
    assert captured[2].shape == (3, 6)


def test_sq_mesh_export_assigns_distinct_materials_per_primitive(tmp_path):
    from autodec.visualizations.sq_mesh import export_sq_mesh

    path = tmp_path / "sq_mesh.obj"

    export_sq_mesh(path, _two_primitive_outdict(), resolution=6)

    obj_text = path.read_text()
    mtl_text = path.with_suffix(".mtl").read_text()
    assert "mtllib sq_mesh.mtl" in obj_text
    assert "usemtl primitive_0000" in obj_text
    assert "usemtl primitive_0001" in obj_text
    assert "newmtl primitive_0000" in mtl_text
    assert "newmtl primitive_0001" in mtl_text
    assert "Kd 0.901961 0.223529 0.274510" in mtl_text
    assert "Kd 0.113725 0.207843 0.341176" in mtl_text


def test_sq_mesh_vertices_clamp_out_of_range_shape_exponents():
    from autodec.visualizations.sq_mesh import _primitive_vertices

    resolution = 6
    eta_idx = 2
    omega_idx = 4
    vertices = _primitive_vertices(
        scale=np.ones(3, dtype=np.float32),
        shape=np.array([0.0, 3.0], dtype=np.float32),
        rotate=np.eye(3, dtype=np.float32),
        trans=np.zeros(3, dtype=np.float32),
        resolution=resolution,
    )

    eta = np.linspace(-np.pi / 2.0, np.pi / 2.0, resolution, dtype=np.float32)[eta_idx]
    omega = np.linspace(-np.pi, np.pi, resolution, endpoint=False, dtype=np.float32)[
        omega_idx
    ]
    expected = np.array(
        [
            np.sign(np.cos(eta)) * np.abs(np.cos(eta)) ** 0.1 * np.cos(omega) ** 2.0,
            np.sign(np.sin(omega)) * np.abs(np.cos(eta)) ** 0.1 * np.abs(np.sin(omega)) ** 2.0,
            np.sign(np.sin(eta)) * np.abs(np.sin(eta)) ** 0.1,
        ],
        dtype=np.float32,
    )

    assert np.allclose(vertices[eta_idx * resolution + omega_idx], expected, atol=1e-6)


def test_visualizations_folder_has_same_name_documentation():
    doc = "autodec/visualizations/visualizations.md"

    with open(doc) as handle:
        text = handle.read()

    assert "AutoDecEpochVisualizer" in text
    assert "data/viz" in text
    assert "WandB" in text
