"""
Recompute F-score metrics for a previously evaluated NPZ file.

The original evaluation script had a bug in get_outdict():
    F = [2 * precision[i] * recall[i] / (precision[i] + precision[i] + 1e-7) ...]
The denominator erroneously used precision twice instead of (precision + recall).
This script loads an existing NPZ + its metrics CSV and overwrites only the
f-score, f-score-15 and f-score-20 columns with the corrected values.
"""

import os
import csv
import random
import shutil

import numpy as np
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

from superdec.utils.predictions_handler_extended import PredictionHandler
from .evaluation import build_dataloader, _build_dataloader, get_outdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: DictConfig):
    print("\n========== SuperDec F-score Recomputation ==========")
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    dataset    = cfg.dataloader.get("dataset", "shapenet")
    source_folder = cfg.get("source_folder", dataset)
    split      = cfg.get(dataset).get("split", "test")
    small      = cfg.get("small", False)
    module_type = cfg.get("type", "iou")

    # Determine extras suffix (must mirror batch_evaluate.py logic)
    extras = ""
    if module_type == "iou":
        if getattr(cfg.optimization, "tapering", True):  extras += "t"
        if getattr(cfg.optimization, "bending",  True):  extras += "b"
        if getattr(cfg.optimization, "reorient",  True): extras += "r"
        if getattr(cfg.optimization, "pruning",   False): extras += "p"
    folder_name = module_type + (f"_{extras}" if extras else "")

    output_dir = os.path.join("data", "output_npz", source_folder, folder_name)
    output_npz  = os.path.join(output_dir, f"{split}.npz")
    if module_type == "none":
        tmp = os.path.join("data", "output_npz", source_folder)
        output_npz  = os.path.join(tmp, f"{dataset}_{split}.npz")
    if small:
        output_dir = output_dir.replace("output_npz", "output_npz/small")
    metrics_csv = os.path.join(output_dir, f"{split}_metrics.csv")

    print(f"Loading NPZ  : {output_npz}")
    print(f"Loading CSV  : {metrics_csv}")

    if not os.path.isfile(output_npz):
        raise FileNotFoundError(f"NPZ not found: {output_npz}")
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    pred_handler = PredictionHandler.from_npz(output_npz)

    # Build dataloader (same as batch_evaluate.py)
    dataloader = build_dataloader(cfg)
    assert dataloader.dataset.normalize == False, "Eval dataset must not be normalised."

    dataset_names = [m["model_id"] for m in dataloader.dataset.models]
    assert np.array_equal(pred_handler.names, dataset_names), (
        "Object order differs between pred_handler and dataset."
    )

    valid_indices = list(range(len(pred_handler.names)))
    if small:
        valid_indices = random.sample(valid_indices, min(len(valid_indices), 128))
        subset = Subset(dataloader.dataset, valid_indices)
        dataloader = _build_dataloader(cfg, subset)

    # Load existing rows (keyed by object index)
    rows_by_index: dict[int, dict] = {}
    fieldnames: list[str] = []
    with open(metrics_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows_by_index[int(row["index"])] = row

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_updated = 0
    old_fscores: dict[int, tuple[float, float, float]] = {
        k: (float(v["f-score"]), float(v["f-score-15"]), float(v["f-score-20"]))
        for k, v in rows_by_index.items()
    }

    for batch in tqdm(dataloader, desc="Recomputing F-scores"):
        batch_indices = batch["idx"]
        points        = batch["points"].to(device)   # (B, N, 3) GT point cloud

        for b_idx in range(len(batch_indices)):
            idx = int(batch_indices[b_idx])

            try:
                mesh = pred_handler.get_mesh(idx, resolution=100, colors=False)
            except Exception as e:
                print(f"  [WARN] mesh generation failed for idx={idx}: {e}")
                continue

            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                f1 = f15 = f20 = 0.0
            else:
                pc_pred = mesh.sample(points.shape[1]).astype(np.float32)
                pc_gt   = points[b_idx].cpu().numpy()
                try:
                    out = get_outdict(pc_gt, None, pc_pred, None)
                    f1  = out["f-score"]
                    f15 = out["f-score-15"]
                    f20 = out["f-score-20"]
                except Exception as e:
                    print(f"  [WARN] F-score computation failed for idx={idx}: {e}")
                    f1 = f15 = f20 = 0.0

            if idx in rows_by_index:
                rows_by_index[idx]["f-score"]    = f1
                rows_by_index[idx]["f-score-15"] = f15
                rows_by_index[idx]["f-score-20"] = f20
                n_updated += 1
            else:
                print(f"  [WARN] idx={idx} not found in existing CSV — skipping.")

    # Write updated CSV back
    old_csv = metrics_csv.replace(".csv", "_old.csv")
    shutil.copy2(metrics_csv, old_csv)
    print(f"Backed up original metrics to {old_csv}")
    print(f"\nWriting updated metrics to {metrics_csv} ({n_updated} rows updated)...")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_by_index.values():
            writer.writerow(row)

    # Print updated aggregate
    all_rows = list(rows_by_index.values())
    count = len(all_rows)
    if count:
        def mean_col(col):
            return sum(float(r[col]) for r in all_rows) / count

        print("\n----- Results (unchanged from CSV) -----")
        print(f"{'mean_chamfer_l1':>25}: {mean_col('chamfer-L1'):.6f}")
        print(f"{'mean_chamfer_l2':>25}: {mean_col('chamfer-L2'):.6f}")
        print(f"{'mean_iou':>25}: {mean_col('iou'):.6f}")
        print(f"{'avg_num_primitives':>25}: {mean_col('num_primitives'):.6f}")
        updated_keys = [k for k in rows_by_index if k in old_fscores]
        old_f1  = sum(old_fscores[k][0] for k in updated_keys) / max(len(updated_keys), 1)
        old_f15 = sum(old_fscores[k][1] for k in updated_keys) / max(len(updated_keys), 1)
        old_f20 = sum(old_fscores[k][2] for k in updated_keys) / max(len(updated_keys), 1)

        print("\n----- F-score: old (buggy) → new (fixed) -----")
        print(f"{'mean_f-score':>25}: {old_f1:.6f}  →  {mean_col('f-score'):.6f}")
        print(f"{'mean_f-score-15':>25}: {old_f15:.6f}  →  {mean_col('f-score-15'):.6f}")
        print(f"{'mean_f-score-20':>25}: {old_f20:.6f}  →  {mean_col('f-score-20'):.6f}")


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../configs", config_name="batch_eval")
    def run_main(cfg: DictConfig):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        main(cfg)

    run_main()
