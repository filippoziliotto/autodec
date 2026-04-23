from pathlib import Path

import numpy as np
import torch


def normalize_points(points):
    translation = points.mean(axis=0)
    normalized = points - translation
    scale = 2 * np.max(np.abs(normalized))
    scale = max(float(scale), 1e-4)
    normalized = normalized / scale
    return normalized.astype(np.float32), translation.astype(np.float32), np.float32(scale)


def _pointcloud_file(model_dir):
    model_dir = Path(model_dir)
    for filename in ("pointcloud_4096.npz", "pointcloud.npz", "points.npz"):
        candidate = model_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No pointcloud file found in {model_dir}")


def load_source_pointcloud(model_dir, n_points=4096):
    pc_data = np.load(_pointcloud_file(model_dir))
    if "points" not in pc_data:
        raise KeyError(f"Pointcloud file in {model_dir} is missing 'points'")
    points = pc_data["points"].astype(np.float32)

    total_points = points.shape[0]
    if total_points >= n_points:
        indices = np.linspace(0, total_points - 1, n_points, dtype=np.int64)
        points = points[indices]
    else:
        indices = np.arange(n_points, dtype=np.int64) % total_points
        points = points[indices]

    normalized, translation, scale = normalize_points(points)
    return {
        "points": torch.from_numpy(normalized),
        "translation": torch.from_numpy(translation),
        "scale": float(scale),
    }
