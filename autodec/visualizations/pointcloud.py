from pathlib import Path

import numpy as np
import torch


def points_to_numpy(points, sample_index=0, max_points=None):
    """Return one point cloud as a CPU numpy array with shape [N, 3]."""

    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    points = np.asarray(points)

    if points.ndim == 3:
        if points.shape[-1] == 3:
            points = points[sample_index]
        elif points.shape[1] == 3:
            points = points[sample_index].T
        else:
            raise ValueError(f"Cannot interpret point tensor with shape {points.shape}")
    elif points.ndim == 2:
        if points.shape[-1] == 3:
            pass
        elif points.shape[0] == 3:
            points = points.T
        else:
            raise ValueError(f"Cannot interpret point tensor with shape {points.shape}")
    else:
        raise ValueError(f"Cannot interpret point tensor with shape {points.shape}")

    points = points.astype(np.float32, copy=False)
    if max_points is not None and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[indices]
    return points


def _colors_for_points(num_points, color):
    color = np.asarray(color, dtype=np.uint8)
    if color.ndim == 1:
        if color.shape[0] == 4:
            color = color[:3]
        if color.shape[0] != 3:
            raise ValueError("Point color must have 3 RGB channels")
        return np.tile(color[None], (num_points, 1))
    if color.shape != (num_points, 3):
        raise ValueError(f"Color array must have shape {(num_points, 3)}, got {color.shape}")
    return color


def write_point_cloud_ply(path, points, color=(210, 210, 210), sample_index=0, max_points=None):
    """Write an ASCII PLY point cloud with per-point RGB colors."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = points_to_numpy(points, sample_index=sample_index, max_points=max_points)
    colors = _colors_for_points(points.shape[0], color)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color_row in zip(points, colors):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color_row[0])} {int(color_row[1])} {int(color_row[2])}\n"
            )
    return path
