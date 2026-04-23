import re
import sys
from pathlib import Path

import numpy as np
import torch

from gendec.config import explicit_config_argument, fallback_cli_config, load_yaml_config


_EPOCH_PATTERN = re.compile(r"epoch_(\d{4,})_preview\.pt$")


def _epoch_from_path(path):
    match = _EPOCH_PATTERN.match(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def collect_preview_epochs(preview_dir, every_n_epochs=10):
    preview_dir = Path(preview_dir)
    selected = []
    for path in sorted(preview_dir.glob("epoch_*_preview.pt")):
        epoch = _epoch_from_path(path)
        if epoch is None:
            continue
        if epoch % int(every_n_epochs) == 0:
            selected.append((epoch, path))
    return selected


def _points_from_preview(path, sample_index=0):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    points = payload["preview_points"]
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float32)
    batch_size = int(points.shape[0])
    if batch_size <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    sample_index = int(sample_index) % batch_size
    points = points[sample_index]
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    mask = np.any(np.abs(points) > 0, axis=1)
    return points[mask]


def _frame_from_points(points, epoch, image_size=(640, 640)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width, height = image_size
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Epoch {epoch}")

    if points.shape[0] > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8, c="#2a9d8f", alpha=0.95)
        limit = max(float(np.abs(points).max()), 1.0)
    else:
        limit = 1.0

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=35)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    plt.close(fig)
    return frame


def _build_single_preview_video(
    preview_dir,
    video_path,
    selected,
    fps=4,
    sample_index=0,
):
    import cv2

    writer = None
    try:
        for epoch, path in selected:
            points = _points_from_preview(path, sample_index=sample_index)
            frame = _frame_from_points(points, epoch)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {video_path}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()

    return video_path


def build_preview_video(
    preview_dir,
    run_name,
    output_root="gendec/videos",
    every_n_epochs=10,
    fps=4,
    sample_index=0,
    num_videos=10,
):
    selected = collect_preview_epochs(preview_dir, every_n_epochs=every_n_epochs)
    if not selected:
        raise FileNotFoundError(f"No preview files found in {preview_dir} matching every_n_epochs={every_n_epochs}")

    output_dir = Path(output_root) / str(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_paths = []
    for video_index in range(int(num_videos)):
        video_path = output_dir / f"video_{video_index:06d}.mp4"
        _build_single_preview_video(
            preview_dir=preview_dir,
            video_path=video_path,
            selected=selected,
            fps=fps,
            sample_index=int(sample_index) + video_index,
        )
        video_paths.append(video_path)

    return {
        "preview_dir": Path(preview_dir),
        "video_paths": video_paths,
        "num_frames": len(selected),
        "epochs": [epoch for epoch, _ in selected],
    }


def _run_from_cfg(cfg):
    preview_cfg = cfg.preview_video
    result = build_preview_video(
        preview_dir=preview_cfg.preview_dir,
        run_name=preview_cfg.run_name,
        output_root=preview_cfg.output_root,
        every_n_epochs=preview_cfg.every_n_epochs,
        fps=preview_cfg.fps,
        sample_index=preview_cfg.sample_index,
        num_videos=preview_cfg.num_videos,
    )
    for path in result["video_paths"]:
        print(path)


if __name__ == "__main__":
    explicit_config = explicit_config_argument("preview_video.yaml")
    if explicit_config is not None:
        _run_from_cfg(load_yaml_config(explicit_config))
    else:
        _run_from_cfg(fallback_cli_config("preview_video.yaml"))
