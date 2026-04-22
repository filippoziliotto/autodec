from pathlib import Path

import numpy as np
import torch


_COLORS = np.array(
    [
        [255,  50,  50, 255],   # red
        [ 50, 180, 255, 255],   # sky blue
        [ 50, 255, 100, 255],   # green
        [255, 220,   0, 255],   # yellow
        [210,  60, 255, 255],   # violet
        [  0, 240, 240, 255],   # cyan
        [255, 130,  20, 255],   # orange
        [255,  80, 200, 255],   # hot pink
        [180, 255,  20, 255],   # lime
        [110, 110, 255, 255],   # periwinkle
        [255, 210,  80, 255],   # amber
        [ 40, 255, 180, 255],   # mint
    ],
    dtype=np.uint8,
)
MIN_SHAPE_EXPONENT = 0.1
MAX_SHAPE_EXPONENT = 2.0


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _signed_power(value, exponent, eps=1e-6):
    return np.sign(value) * np.maximum(np.abs(value), eps) ** exponent


def _active_mask(outdict, sample_index, exist_threshold):
    if "exist" in outdict:
        exist = _to_numpy(outdict["exist"])[sample_index, :, 0]
    else:
        logits = _to_numpy(outdict["exist_logit"])[sample_index, :, 0]
        exist = 1.0 / (1.0 + np.exp(-logits))
    return exist > exist_threshold


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


def build_sq_mesh(outdict, sample_index=0, resolution=24, exist_threshold=0.5):
    """Build a combined trimesh mesh for active predicted superquadrics."""

    import trimesh

    scale = _to_numpy(outdict["scale"])[sample_index]
    shape = _to_numpy(outdict["shape"])[sample_index]
    rotate = _to_numpy(outdict["rotate"])[sample_index]
    trans = _to_numpy(outdict["trans"])[sample_index]
    active = _active_mask(outdict, sample_index, exist_threshold)
    faces = _grid_faces(resolution, resolution)

    meshes = []
    for primitive_idx in np.flatnonzero(active):
        vertices = _primitive_vertices(
            scale[primitive_idx],
            shape[primitive_idx],
            rotate[primitive_idx],
            trans[primitive_idx],
            resolution,
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        color = _COLORS[primitive_idx % len(_COLORS)]
        mesh.visual.face_colors = np.tile(color[None], (faces.shape[0], 1))
        meshes.append(mesh)

    if not meshes:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _write_mesh_ply(path, mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_colors = np.asarray(getattr(mesh.visual, "face_colors", []), dtype=np.uint8)
    has_face_colors = face_colors.shape[0] == faces.shape[0] and face_colors.shape[1] >= 3

    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        if has_face_colors:
            handle.write("property uchar red\n")
            handle.write("property uchar green\n")
            handle.write("property uchar blue\n")
            handle.write("property uchar alpha\n")
        handle.write("end_header\n")

        for vertex in vertices:
            handle.write(f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face_idx, face in enumerate(faces):
            line = f"3 {face[0]} {face[1]} {face[2]}"
            if has_face_colors:
                color = face_colors[face_idx]
                alpha = color[3] if color.shape[0] > 3 else 255
                line += f" {color[0]} {color[1]} {color[2]} {alpha}"
            handle.write(f"{line}\n")


def _write_mesh_obj(path, mesh):
    path = Path(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_colors = np.asarray(getattr(mesh.visual, "face_colors", []), dtype=np.uint8)
    has_face_colors = face_colors.shape[0] == faces.shape[0] and face_colors.shape[1] >= 3
    material_by_color = {}
    materials = []
    if has_face_colors:
        for color in face_colors:
            key = tuple(int(channel) for channel in color[:4])
            if key not in material_by_color:
                material_by_color[key] = f"primitive_{len(materials):04d}"
                materials.append((material_by_color[key], key))

    if materials:
        mtl_path = path.with_suffix(".mtl")
        with mtl_path.open("w", encoding="utf-8") as handle:
            handle.write("# AutoDec superquadric scaffold materials\n")
            for name, color in materials:
                red, green, blue = (channel / 255.0 for channel in color[:3])
                alpha = color[3] / 255.0 if len(color) > 3 else 1.0
                handle.write(f"newmtl {name}\n")
                handle.write(f"Ka {red:.6f} {green:.6f} {blue:.6f}\n")
                handle.write(f"Kd {red:.6f} {green:.6f} {blue:.6f}\n")
                handle.write(f"Ke {red:.6f} {green:.6f} {blue:.6f}\n")
                handle.write(f"d {alpha:.6f}\n\n")

    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("# AutoDec superquadric scaffold\n")
        if materials:
            handle.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        current_material = None
        for face_idx, face in enumerate(faces):
            if has_face_colors:
                key = tuple(int(channel) for channel in face_colors[face_idx, :4])
                material = material_by_color[key]
                if material != current_material:
                    handle.write(f"usemtl {material}\n")
                    current_material = material
            # OBJ indices are 1-based.
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def export_sq_mesh(path, outdict, sample_index=0, resolution=24, exist_threshold=0.5):
    """Export active predicted superquadrics as a mesh."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = build_sq_mesh(
        outdict,
        sample_index=sample_index,
        resolution=resolution,
        exist_threshold=exist_threshold,
    )
    if path.suffix.lower() == ".ply":
        _write_mesh_ply(path, mesh)
    elif path.suffix.lower() == ".obj":
        _write_mesh_obj(path, mesh)
    else:
        mesh.export(path, file_type=path.suffix.lstrip(".") or "ply")
    return path
