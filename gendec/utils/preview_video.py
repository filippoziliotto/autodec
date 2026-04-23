import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

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


def ensure_matplotlib_cache_dir(output_root):
    cache_dir = Path(output_root) / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    return cache_dir


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


def _write_frames(selected, frame_dir, sample_index=0):
    from PIL import Image

    frame_paths = []
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, (epoch, path) in enumerate(selected):
        points = _points_from_preview(path, sample_index=sample_index)
        frame = _frame_from_points(points, epoch)
        frame_path = frame_dir / f"frame_{frame_index:06d}.png"
        Image.fromarray(frame, mode="RGB").save(frame_path)
        frame_paths.append(frame_path)
    return frame_paths


def _ffmpeg_command(frame_dir, video_path, fps):
    return [
        shutil.which("ffmpeg") or "ffmpeg",
        "-y",
        "-framerate",
        str(float(fps)),
        "-i",
        str(Path(frame_dir) / "frame_%06d.png"),
        "-c:v",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]


def _encode_video_with_ffmpeg(frame_dir, video_path, fps=4):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    command = _ffmpeg_command(frame_dir, video_path, fps=fps)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr}")
    return True


def _build_single_preview_video(
    video_path,
    selected,
    fps=4,
    sample_index=0,
):
    frame_dir = Path(video_path).with_suffix("")
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    try:
        frame_paths = _write_frames(selected, frame_dir, sample_index=sample_index)
        try:
            if _encode_video_with_ffmpeg(frame_dir, video_path, fps=fps):
                return video_path, "ffmpeg"
        except Exception:
            pass

        import cv2

        writer = None
        try:
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Could not read rendered frame {frame_path}")
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
                writer.write(frame)
        finally:
            if writer is not None:
                writer.release()
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    return video_path, "opencv"


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
    ensure_matplotlib_cache_dir(output_dir)
    video_paths = []
    backend = None
    for video_index in range(int(num_videos)):
        video_path = output_dir / f"video_{video_index:06d}.mp4"
        video_path, current_backend = _build_single_preview_video(
            video_path=video_path,
            selected=selected,
            fps=fps,
            sample_index=int(sample_index) + video_index,
        )
        video_paths.append(video_path)
        if backend is None:
            backend = current_backend

    return {
        "preview_dir": Path(preview_dir),
        "video_paths": video_paths,
        "num_frames": len(selected),
        "epochs": [epoch for epoch, _ in selected],
        "backend": backend,
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
