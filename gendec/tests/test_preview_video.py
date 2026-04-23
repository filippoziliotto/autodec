from pathlib import Path

import torch


def _write_preview(path, points_scale):
    path.parent.mkdir(parents=True, exist_ok=True)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ) * float(points_scale)
    torch.save(
        {
            "tokens": torch.zeros(1, 16, 15),
            "exist": torch.zeros(1, 16, 1),
            "active_mask": torch.zeros(1, 16, dtype=torch.bool),
            "preview_points": points.unsqueeze(0),
        },
        path,
    )


def test_collect_preview_epochs_selects_every_10(tmp_path):
    from gendec.utils.preview_video import collect_preview_epochs

    preview_dir = tmp_path / "previews"
    for epoch in [0, 1, 9, 10, 11, 20]:
        _write_preview(preview_dir / f"epoch_{epoch:04d}_preview.pt", points_scale=epoch + 1)

    selected = collect_preview_epochs(preview_dir, every_n_epochs=10)

    assert [item[0] for item in selected] == [0, 10, 20]


def test_build_preview_video_writes_video_under_run_name(tmp_path):
    from gendec.utils.preview_video import build_preview_video

    preview_dir = tmp_path / "previews"
    _write_preview(preview_dir / "epoch_0000_preview.pt", points_scale=1.0)
    _write_preview(preview_dir / "epoch_0010_preview.pt", points_scale=2.0)
    output_root = tmp_path / "videos"

    result = build_preview_video(
        preview_dir=preview_dir,
        run_name="gendec_run",
        output_root=output_root,
        every_n_epochs=10,
        fps=2,
    )

    assert result["num_frames"] == 2
    assert result["video_paths"][0] == output_root / "gendec_run" / "video_000000.mp4"
    assert result["video_paths"][9] == output_root / "gendec_run" / "video_000009.mp4"
    assert len(result["video_paths"]) == 10
    assert all(path.is_file() for path in result["video_paths"])
