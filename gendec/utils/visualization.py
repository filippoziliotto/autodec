import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


_COLORS = np.array(
    [
        [255, 50, 50, 255],
        [50, 180, 255, 255],
        [50, 255, 100, 255],
        [255, 220, 0, 255],
        [210, 60, 255, 255],
        [0, 240, 240, 255],
        [255, 130, 20, 255],
        [255, 80, 200, 255],
        [180, 255, 20, 255],
        [110, 110, 255, 255],
        [255, 210, 80, 255],
        [40, 255, 180, 255],
    ],
    dtype=np.uint8,
)
MIN_SHAPE_EXPONENT = 0.1
MAX_SHAPE_EXPONENT = 2.0


@dataclass(frozen=True)
class GeneratedSQVisualizationRecord:
    split: str
    sample_index: int
    sample_dir: Path
    sq_mesh_path: Path
    preview_points_path: Path
    decoded_points_path: Path | None
    metadata_path: Path


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sample_points(value, sample_index):
    if isinstance(value, (list, tuple)):
        sample = value[sample_index]
        if torch.is_tensor(sample):
            return sample.detach().cpu().numpy()
        return np.asarray(sample)
    return _to_numpy(value)[sample_index]


def _signed_power(value, exponent, eps=1e-6):
    return np.sign(value) * np.maximum(np.abs(value), eps) ** exponent


def _grid_faces(n_eta, n_omega):
    faces = []
    for eta_idx in range(n_eta - 1):
        for omega_idx in range(n_omega):
            next_omega = (omega_idx + 1) % n_omega
            v00 = eta_idx * n_omega + omega_idx
            v01 = eta_idx * n_omega + next_omega
            v10 = (eta_idx + 1) * n_omega + omega_idx
            v11 = (eta_idx + 1) * n_omega + next_omega
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return np.asarray(faces, dtype=np.int64)


def _primitive_vertices(scale, shape, rotate, trans, resolution):
    shape = np.clip(shape, MIN_SHAPE_EXPONENT, MAX_SHAPE_EXPONENT)
    etas = np.linspace(-np.pi / 2.0, np.pi / 2.0, resolution, dtype=np.float32)
    omegas = np.linspace(-np.pi, np.pi, resolution, endpoint=False, dtype=np.float32)
    eta_grid, omega_grid = np.meshgrid(etas, omegas, indexing="ij")

    e1, e2 = shape
    sx, sy, sz = scale
    x = sx * _signed_power(np.cos(eta_grid), e1) * _signed_power(np.cos(omega_grid), e2)
    y = sy * _signed_power(np.cos(eta_grid), e1) * _signed_power(np.sin(omega_grid), e2)
    z = sz * _signed_power(np.sin(eta_grid), e1)

    canonical = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return canonical @ rotate.T + trans


def _write_mesh_obj(path, vertices, faces, face_colors):
    path = Path(path)
    material_by_color = {}
    materials = []
    for color in face_colors:
        key = tuple(int(channel) for channel in color[:4])
        if key not in material_by_color:
            material_by_color[key] = f"primitive_{len(materials):04d}"
            materials.append((material_by_color[key], key))

    if materials:
        mtl_path = path.with_suffix(".mtl")
        with mtl_path.open("w", encoding="utf-8") as handle:
            handle.write("# GenDec generated SQ materials\n")
            for name, color in materials:
                red, green, blue = (channel / 255.0 for channel in color[:3])
                alpha = color[3] / 255.0 if len(color) > 3 else 1.0
                handle.write(f"newmtl {name}\n")
                handle.write(f"Ka {red:.6f} {green:.6f} {blue:.6f}\n")
                handle.write(f"Kd {red:.6f} {green:.6f} {blue:.6f}\n")
                handle.write(f"Ke {red:.6f} {green:.6f} {blue:.6f}\n")
                handle.write(f"d {alpha:.6f}\n\n")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# GenDec generated superquadric scaffold\n")
        if materials:
            handle.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        current_material = None
        for face_idx, face in enumerate(faces):
            key = tuple(int(channel) for channel in face_colors[face_idx, :4])
            material = material_by_color[key]
            if material != current_material:
                handle.write(f"usemtl {material}\n")
                current_material = material
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def write_point_cloud_ply(path, points, color=(210, 210, 210), max_points=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = _to_numpy(points).astype(np.float32, copy=False)
    if max_points is not None and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[indices]
    colors = np.tile(np.asarray(color, dtype=np.uint8)[None], (points.shape[0], 1))
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


class GeneratedSQVisualizer:
    def __init__(
        self,
        root_dir="data/viz",
        run_name="gendec_eval",
        mesh_resolution=24,
        exist_threshold=0.5,
        max_preview_points=4096,
    ):
        self.root_dir = Path(root_dir)
        self.run_name = run_name
        self.mesh_resolution = int(mesh_resolution)
        self.exist_threshold = float(exist_threshold)
        self.max_preview_points = int(max_preview_points)

    def _sample_dir(self, split, sample_index):
        return self.root_dir / self.run_name / split / f"generated_{sample_index:04d}"

    def _write_metadata(self, path, split, sample_index, preview_points, active_primitives, decoded_points=None):
        payload = {
            "split": split,
            "sample_index": int(sample_index),
            "preview_points": int(preview_points),
            "active_primitives": int(active_primitives),
        }
        if decoded_points is not None:
            payload["decoded_points"] = int(decoded_points)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _export_sq_mesh(self, path, processed, sample_index):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scale = _to_numpy(processed["scale"])[sample_index]
        shape = _to_numpy(processed["shape"])[sample_index]
        rotate = _to_numpy(processed["rotate"])[sample_index]
        trans = _to_numpy(processed["trans"])[sample_index]
        active = _to_numpy(processed["active_mask"])[sample_index].astype(bool)
        faces = _grid_faces(self.mesh_resolution, self.mesh_resolution)

        all_vertices = []
        all_faces = []
        all_face_colors = []
        vertex_offset = 0
        for primitive_idx in np.flatnonzero(active):
            vertices = _primitive_vertices(
                scale[primitive_idx],
                shape[primitive_idx],
                rotate[primitive_idx],
                trans[primitive_idx],
                self.mesh_resolution,
            )
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            all_face_colors.append(np.tile(_COLORS[primitive_idx % len(_COLORS)][None], (faces.shape[0], 1)))
            vertex_offset += vertices.shape[0]

        if all_vertices:
            vertices = np.concatenate(all_vertices, axis=0)
            mesh_faces = np.concatenate(all_faces, axis=0)
            face_colors = np.concatenate(all_face_colors, axis=0)
        else:
            vertices = np.zeros((0, 3), dtype=np.float32)
            mesh_faces = np.zeros((0, 3), dtype=np.int64)
            face_colors = np.zeros((0, 4), dtype=np.uint8)

        _write_mesh_obj(path, vertices, mesh_faces, face_colors)
        return path

    def write_generated(self, processed, split="test", num_samples=10, decoded_points=None):
        batch_size = int(processed["tokens"].shape[0])
        num_samples = min(int(num_samples), batch_size)
        records = []

        for sample_index in range(num_samples):
            sample_dir = self._sample_dir(split, sample_index)
            sample_dir.mkdir(parents=True, exist_ok=True)

            sq_mesh_path = sample_dir / "sq_mesh.obj"
            preview_points_path = sample_dir / "preview_points.ply"
            decoded_points_path = None
            metadata_path = sample_dir / "metadata.json"

            self._export_sq_mesh(sq_mesh_path, processed, sample_index)
            preview_points = _to_numpy(processed["preview_points"])[sample_index]
            non_zero = np.any(np.abs(preview_points) > 0, axis=1)
            preview_points = preview_points[non_zero]
            write_point_cloud_ply(
                preview_points_path,
                preview_points,
                color=(42, 157, 143),
                max_points=self.max_preview_points,
            )
            decoded_count = None
            if decoded_points is not None:
                decoded_points_path = sample_dir / "decoded_points.ply"
                decoded_sample = _sample_points(decoded_points, sample_index)
                non_zero = np.any(np.abs(decoded_sample) > 0, axis=1)
                decoded_sample = decoded_sample[non_zero]
                write_point_cloud_ply(
                    decoded_points_path,
                    decoded_sample,
                    color=(231, 111, 81),
                    max_points=self.max_preview_points,
                )
                decoded_count = decoded_sample.shape[0]
            self._write_metadata(
                metadata_path,
                split,
                sample_index,
                preview_points.shape[0],
                int(_to_numpy(processed["active_mask"])[sample_index].sum()),
                decoded_points=decoded_count,
            )
            records.append(
                GeneratedSQVisualizationRecord(
                    split=split,
                    sample_index=sample_index,
                    sample_dir=sample_dir,
                    sq_mesh_path=sq_mesh_path,
                    preview_points_path=preview_points_path,
                    decoded_points_path=decoded_points_path,
                    metadata_path=metadata_path,
                )
            )

        return records
