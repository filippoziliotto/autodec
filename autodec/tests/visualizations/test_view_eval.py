import json

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
    assert samples[0].reconstruction_path == sample_dir / "reconstruction.ply"
    assert samples[0].gt_path == sample_dir / "input_gt.ply"
    assert samples[0].metadata_path == sample_dir / "metadata.json"


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
    assert args.reconstruction_port == 8092
    assert args.gt_port == 8093


def test_render_wrapper_html_has_three_iframes_and_navigation():
    from autodec.visualizations.view_eval import render_wrapper_html

    html = render_wrapper_html(
        title="AutoDec Viewer",
        pane_urls={
            "sq": "http://127.0.0.1:8091",
            "reconstruction": "http://127.0.0.1:8092",
            "gt": "http://127.0.0.1:8093",
        },
    )

    assert "AutoDec Viewer" in html
    assert "Superquadric reconstruction" in html
    assert "Point reconstruction" in html
    assert "Ground truth" in html
    assert "http://127.0.0.1:8091" in html
    assert "http://127.0.0.1:8092" in html
    assert "http://127.0.0.1:8093" in html
    assert "Previous" in html
    assert "Next" in html
    assert "/api/sample/previous" in html
    assert "/api/sample/next" in html
