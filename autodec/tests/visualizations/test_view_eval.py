import json
from types import SimpleNamespace

import pytest


def _write_sample(path, metadata=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "sq_mesh.obj").write_text("o mesh\n", encoding="utf-8")
    (path / "reconstruction.ply").write_text("ply\n", encoding="utf-8")
    (path / "input_gt.ply").write_text("ply\n", encoding="utf-8")
    if metadata is not None:
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_discover_samples_accepts_direct_sample_path(tmp_path):
    from autodec.visualizations.view_eval import discover_samples

    sample_dir = _write_sample(tmp_path / "sample_0000", {"category": "chair"})

    samples = discover_samples(sample_dir)

    assert [sample.sample_dir for sample in samples] == [sample_dir]
    assert samples[0].sq_mesh_path == sample_dir / "sq_mesh.obj"
    assert samples[0].sq_mesh_lm_path is None
    assert samples[0].reconstruction_path == sample_dir / "reconstruction.ply"
    assert samples[0].gt_path == sample_dir / "input_gt.ply"
    assert samples[0].metadata_path == sample_dir / "metadata.json"


def test_discover_samples_records_optional_lm_sq_mesh(tmp_path):
    from autodec.visualizations.view_eval import discover_samples

    sample_dir = _write_sample(tmp_path / "sample_0000")
    (sample_dir / "sq_mesh_lm.obj").write_text("o lm_mesh\n", encoding="utf-8")

    samples = discover_samples(sample_dir)

    assert samples[0].sq_mesh_lm_path == sample_dir / "sq_mesh_lm.obj"


def test_discover_samples_recurses_and_natural_sorts_valid_samples(tmp_path):
    from autodec.visualizations.view_eval import discover_samples

    root = tmp_path / "eval_run"
    late = _write_sample(root / "test" / "epoch_0000" / "sample_0010")
    early = _write_sample(root / "test" / "epoch_0000" / "sample_0002")
    _write_sample(root / "test" / "epoch_0001" / "sample_0000")
    incomplete = root / "test" / "epoch_0000" / "sample_0001"
    incomplete.mkdir(parents=True)
    (incomplete / "sq_mesh.obj").write_text("o mesh\n", encoding="utf-8")

    samples = discover_samples(root)

    assert [sample.sample_dir for sample in samples] == [
        early,
        late,
        root / "test" / "epoch_0001" / "sample_0000",
    ]


def test_discover_samples_errors_when_no_complete_samples_exist(tmp_path):
    from autodec.visualizations.view_eval import discover_samples

    (tmp_path / "sample_0000").mkdir()

    with pytest.raises(ValueError, match="No complete AutoDec visualization samples"):
        discover_samples(tmp_path)


def test_load_sample_metadata_returns_json_or_empty_dict(tmp_path):
    from autodec.visualizations.view_eval import discover_samples, load_sample_metadata

    with_metadata = _write_sample(tmp_path / "sample_0000", {"category": "chair", "active": 3})
    without_metadata = _write_sample(tmp_path / "sample_0001")

    samples = discover_samples(tmp_path)
    by_dir = {sample.sample_dir: sample for sample in samples}

    assert load_sample_metadata(by_dir[with_metadata]) == {"category": "chair", "active": 3}
    assert load_sample_metadata(by_dir[without_metadata]) == {}


def test_cli_parser_imports_without_viser_and_parses_path(tmp_path):
    from autodec.visualizations.view_eval import build_arg_parser

    args = build_arg_parser().parse_args([str(tmp_path), "--host", "127.0.0.1"])

    assert args.visualization_output_path == tmp_path
    assert args.host == "127.0.0.1"
    assert args.wrapper_port == 8090
    assert args.sq_port == 8091
    assert args.lm_sq_port == 8094
    assert args.reconstruction_port == 8092
    assert args.gt_port == 8093


def test_viewer_ports_are_named_and_unique(tmp_path):
    from autodec.visualizations.view_eval import _viewer_ports, build_arg_parser

    args = build_arg_parser().parse_args([str(tmp_path), "--host", "127.0.0.1"])

    assert _viewer_ports(args) == {
        "wrapper": 8090,
        "original SQ": 8091,
        "LM SQ": 8094,
        "reconstruction": 8092,
        "ground truth": 8093,
    }


def test_viewer_port_preflight_rejects_occupied_port(tmp_path, monkeypatch):
    from autodec.visualizations.view_eval import (
        _assert_viewer_ports_available,
        build_arg_parser,
    )

    occupied_port = 49152
    args = build_arg_parser().parse_args(
        [
            str(tmp_path),
            "--host",
            "127.0.0.1",
            "--gt-port",
            str(occupied_port),
        ]
    )

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, timeout):
            self.timeout = timeout

        def connect_ex(self, address):
            return 0 if address[1] == occupied_port else 1

    monkeypatch.setattr(
        "autodec.visualizations.view_eval.socket.socket",
        lambda *args, **kwargs: FakeSocket(),
    )

    with pytest.raises(RuntimeError, match=f"ground truth={occupied_port}"):
        _assert_viewer_ports_available(args)


def test_viewer_port_preflight_rejects_duplicate_ports(tmp_path):
    from autodec.visualizations.view_eval import (
        _assert_viewer_ports_available,
        build_arg_parser,
    )

    args = build_arg_parser().parse_args(
        [
            str(tmp_path),
            "--host",
            "127.0.0.1",
            "--sq-port",
            "8092",
            "--reconstruction-port",
            "8092",
        ]
    )

    with pytest.raises(ValueError, match="duplicate viewer ports"):
        _assert_viewer_ports_available(args)


def test_render_wrapper_html_has_four_iframes_and_navigation():
    from autodec.visualizations.view_eval import render_wrapper_html

    html = render_wrapper_html(
        title="AutoDec Viewer",
        pane_urls={
            "sq": "http://127.0.0.1:8091",
            "lm_sq": "http://127.0.0.1:8094",
            "reconstruction": "http://127.0.0.1:8092",
            "gt": "http://127.0.0.1:8093",
        },
    )

    assert "AutoDec Viewer" in html
    assert html.count("<iframe") == 4
    assert "Original SQ" in html
    assert "LM optimized SQ" in html
    assert "Point reconstruction" in html
    assert "Ground truth" in html
    assert "http://127.0.0.1:8091" in html
    assert "http://127.0.0.1:8094" in html
    assert "http://127.0.0.1:8092" in html
    assert "http://127.0.0.1:8093" in html
    assert "Previous" in html
    assert "Next" in html
    assert "/api/sample/previous" in html
    assert "/api/sample/next" in html


def test_obj_material_face_colors_parse_per_primitive_colors(tmp_path):
    from autodec.visualizations.view_eval import _obj_material_face_colors

    obj_path = tmp_path / "sq_mesh.obj"
    obj_path.write_text(
        "\n".join(
            [
                "mtllib sq_mesh.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "usemtl primitive_0000",
                "f 1 2 3",
                "usemtl primitive_0001",
                "f 1 3 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "sq_mesh.mtl").write_text(
        "\n".join(
            [
                "newmtl primitive_0000",
                "Kd 1.000000 0.000000 0.000000",
                "d 1.000000",
                "newmtl primitive_0001",
                "Kd 0.000000 0.500000 1.000000",
                "d 0.750000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    face_colors = _obj_material_face_colors(obj_path, face_count=2)

    assert face_colors.tolist() == [
        [255, 0, 0, 255],
        [0, 128, 255, 191],
    ]


def test_viser_pane_converts_obj_materials_to_color_visuals(tmp_path):
    import trimesh
    from trimesh.visual.color import ColorVisuals

    from autodec.visualizations.view_eval import _ViserPane

    obj_path = tmp_path / "sq_mesh.obj"
    obj_path.write_text(
        "\n".join(
            [
                "mtllib sq_mesh.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "usemtl primitive_0000",
                "f 1 2 3",
                "usemtl primitive_0001",
                "f 1 3 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "sq_mesh.mtl").write_text(
        "\n".join(
            [
                "newmtl primitive_0000",
                "Kd 1.000000 0.000000 0.000000",
                "d 1.000000",
                "newmtl primitive_0001",
                "Kd 0.000000 0.500000 1.000000",
                "d 1.000000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    captured = {}

    class FakeScene:
        def add_mesh_trimesh(self, name, mesh, visible):
            captured["name"] = name
            captured["mesh"] = mesh
            captured["visible"] = visible

    pane = _ViserPane(
        server=SimpleNamespace(scene=FakeScene()),
        pane_type="sq",
        trimesh_module=trimesh,
        point_size=0.005,
    )

    pane._load_mesh(obj_path, "/original_superquadric")

    assert captured["name"] == "/original_superquadric"
    assert captured["visible"] is True
    assert isinstance(captured["mesh"].visual, ColorVisuals)
    assert captured["mesh"].visual.face_colors.tolist() == [
        [255, 0, 0, 255],
        [0, 128, 255, 255],
    ]


def test_viser_pane_forces_old_transparent_obj_materials_opaque(tmp_path):
    import trimesh

    from autodec.visualizations.view_eval import _ViserPane

    obj_path = tmp_path / "sq_mesh.obj"
    obj_path.write_text(
        "\n".join(
            [
                "mtllib sq_mesh.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "usemtl primitive_0000",
                "f 1 2 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "sq_mesh.mtl").write_text(
        "\n".join(
            [
                "newmtl primitive_0000",
                "Kd 1.000000 0.000000 0.000000",
                "d 0.500000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    captured = {}

    class FakeScene:
        def add_mesh_trimesh(self, name, mesh, visible):
            captured["mesh"] = mesh

    pane = _ViserPane(
        server=SimpleNamespace(scene=FakeScene()),
        pane_type="sq",
        trimesh_module=trimesh,
        point_size=0.005,
    )

    pane._load_mesh(obj_path, "/original_superquadric")

    assert captured["mesh"].visual.face_colors.tolist() == [[255, 0, 0, 255]]


def test_viser_panes_route_to_distinct_sample_files(tmp_path):
    from autodec.visualizations.view_eval import EvalVisualizationSample, _ViserPane

    sample = EvalVisualizationSample(
        sample_dir=tmp_path,
        sq_mesh_path=tmp_path / "sq_mesh.obj",
        sq_mesh_lm_path=tmp_path / "sq_mesh_lm.obj",
        reconstruction_path=tmp_path / "reconstruction.ply",
        gt_path=tmp_path / "input_gt.ply",
        metadata_path=tmp_path / "metadata.json",
    )
    calls = []

    class CapturePane(_ViserPane):
        def __init__(self, pane_type):
            super().__init__(
                server=SimpleNamespace(scene=SimpleNamespace()),
                pane_type=pane_type,
                trimesh_module=SimpleNamespace(),
                point_size=0.005,
            )

        def _load_mesh(self, path, name):
            calls.append((self.pane_type, "mesh", path.name, name))

        def _load_point_cloud(self, path, name):
            calls.append((self.pane_type, "points", path.name, name))

    for pane_type in ["sq", "lm_sq", "reconstruction", "gt"]:
        CapturePane(pane_type).load(sample)

    assert calls == [
        ("sq", "mesh", "sq_mesh.obj", "/original_superquadric"),
        ("lm_sq", "mesh", "sq_mesh_lm.obj", "/lm_optimized_superquadric"),
        ("reconstruction", "points", "reconstruction.ply", "/point_reconstruction"),
        ("gt", "points", "input_gt.ply", "/ground_truth"),
    ]
