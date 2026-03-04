"""
Compute superquadric overlap from an NPZ file.

Overlap is defined as the percentage of GT-occupied points
(inside the ground-truth shape) that are contained in more
than one superquadric.

Query points are loaded from points.npz for each model, found by
scanning the ShapeNet data root for the matching category folder.

Usage:
    python -m superoptim.compute_overlap path/to/file.npz --data_root /path/to/shapenet
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from superdec.utils.predictions_handler_extended import PredictionHandler
from .evaluation import sdfs_from_pred_handler


def build_model_path_map(data_root: str) -> dict[str, str]:
    """Scan data_root/category/model_id and return {model_id: full_path}."""
    mapping = {}
    for category in os.listdir(data_root):
        category_path = os.path.join(data_root, category)
        if not os.path.isdir(category_path):
            continue
        for model_id in os.listdir(category_path):
            model_path = os.path.join(category_path, model_id)
            if os.path.isdir(model_path):
                mapping[model_id] = model_path
    return mapping


def load_points_npz(model_path: str):
    """Load points.npz and return (points, occupancy_bool_mask)."""
    path = os.path.join(model_path, "points.npz")
    data = np.load(path)
    points = data["points"].astype(np.float32)
    occ = data["occupancies"]
    if np.issubdtype(occ.dtype, np.uint8):
        occ = np.unpackbits(occ)[:points.shape[0]]
    return points, occ.astype(bool)


def compute_overlap_from_npz(
    npz_path: str,
    data_root: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> tuple[float, np.ndarray]:
    """Compute per-object overlap and mean overlap from an NPZ file.

    Overlap = fraction of GT-occupied points that are inside >1 primitive.

    Args:
        npz_path: path to the NPZ file produced by batch_evaluate.py
        data_root: ShapeNet data root (contains category sub-folders)
        device: torch device string
        batch_size: number of objects to process at once

    Returns:
        mean_overlap: scalar mean overlap across all objects
        per_object_overlap: (N,) array of per-object overlap values
    """
    pred_handler = PredictionHandler.from_npz(npz_path)
    n_objects = pred_handler.scale.shape[0]
    print(f"Loaded {n_objects} objects from {npz_path}")

    print(f"Scanning data root: {data_root} ...")
    model_path_map = build_model_path_map(data_root)
    print(f"Found {len(model_path_map)} models in data root.")

    per_object_overlap       = np.zeros(n_objects, dtype=np.float32)
    per_object_overlap_count  = np.zeros(n_objects, dtype=np.int64)   # # GT-occ points in >1 primitive
    per_object_occupied_count = np.zeros(n_objects, dtype=np.int64)   # total # GT-occ points
    skipped = 0

    for start in tqdm(range(0, n_objects, batch_size), desc="Computing overlap"):
        batch_indices = list(range(start, min(start + batch_size, n_objects)))

        # Load points.npz for each object; skip if not found
        batch_points = []
        batch_occ = []
        valid_indices = []
        for idx in batch_indices:
            model_id = pred_handler.names[idx]
            model_path = model_path_map.get(model_id)
            if model_path is None:
                skipped += 1
                continue
            try:
                pts, occ = load_points_npz(model_path)
            except Exception as e:
                print(f"  [WARN] failed to load points.npz for {model_id}: {e}")
                skipped += 1
                continue
            batch_points.append(pts)
            batch_occ.append(occ)
            valid_indices.append(idx)

        if not valid_indices:
            continue

        # Pad to the same number of points (use min across batch)
        M = min(p.shape[0] for p in batch_points)
        batch_points = np.stack([p[:M] for p in batch_points], axis=0)  # (B, M, 3)
        batch_occ    = np.stack([o[:M] for o in batch_occ],    axis=0)  # (B, M)

        points_t = torch.tensor(batch_points, dtype=torch.float32, device=device)
        occ_t    = torch.tensor(batch_occ,    dtype=torch.bool,    device=device)

        with torch.no_grad():
            sdfs = sdfs_from_pred_handler(pred_handler, valid_indices, points_t, device=device)
            # sdfs: (B, N, M)
            inside_count = (sdfs < 0).sum(dim=1)  # (B, M) — how many primitives contain each point

            for b, idx in enumerate(valid_indices):
                gt_occ_mask = occ_t[b]  # (M,) — GT-occupied points
                if gt_occ_mask.any():
                    overlapping = (inside_count[b][gt_occ_mask] > 1)
                    per_object_overlap_count[idx]  = overlapping.sum().cpu().item()
                    per_object_occupied_count[idx] = gt_occ_mask.sum().cpu().item()
                    per_object_overlap[idx] = overlapping.float().mean().cpu().item()

    if skipped:
        print(f"[WARN] Skipped {skipped} objects (points.npz not found or failed to load).")

    mean_overlap = per_object_overlap.mean()
    return mean_overlap, per_object_overlap, per_object_overlap_count, per_object_occupied_count


def main():
    parser = argparse.ArgumentParser(description="Compute superquadric overlap from an NPZ file.")
    parser.add_argument("npz", help="Path to the NPZ file.")
    parser.add_argument("--data_root", default="data/ShapeNet", help="ShapeNet data root directory.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    mean_overlap, per_object_overlap, overlap_count, occupied_count = compute_overlap_from_npz(
        args.npz, args.data_root, device=args.device, batch_size=args.batch_size
    )

    total_overlap   = int(overlap_count.sum())
    total_occupied  = int(occupied_count.sum())

    print(f"\n----- Overlap Results -----")
    print(f"{'mean_overlap':>25}: {mean_overlap:.6f}  ({mean_overlap * 100:.2f}%)")
    print(f"{'min_overlap':>25}: {per_object_overlap.min():.6f}")
    print(f"{'max_overlap':>25}: {per_object_overlap.max():.6f}")
    print(f"{'std_overlap':>25}: {per_object_overlap.std():.6f}")
    print(f"{'total_overlap_pts':>25}: {total_overlap} / {total_occupied}  ({100 * total_overlap / max(total_occupied, 1):.2f}%)")
    print(f"{'avg_overlap_pts/obj':>25}: {overlap_count.mean():.1f} / {occupied_count.mean():.1f}")


if __name__ == "__main__":
    main()
