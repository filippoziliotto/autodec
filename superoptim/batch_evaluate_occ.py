import os
import torch
import numpy as np
from tqdm import tqdm
from superdec.utils.predictions_handler_extended import PredictionHandler
import viser
import random
import hydra
from omegaconf import DictConfig
import importlib
import gc
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

from superoptim.none.batch_superq import BatchSuperQMulti
from superdec.data.dataloader import ShapeNet, ABO
from .evaluation import get_outdict, eval_mesh, build_dataloader, _build_dataloader, compute_ious_sdf_from_handler

def main(cfg: DictConfig):
    print("\n========== SuperDec Evaluation ==========")
    print("Config:\n" + OmegaConf.to_yaml(cfg))
    
    module_type = 'none'
    dataset = cfg.dataloader.get('dataset', 'shapenet')
    source_folder = cfg.get('source_folder', dataset)
    split = cfg.get(dataset)['split']
    small = cfg.get('small', False)

    print(f"Evaluating {module_type} on {dataset} {split}...")
    input_npz = f"data/output_npz/{source_folder}/{dataset}_{split}.npz"
    # Determine extras suffix (used for folder naming)
    extras = ""
    if module_type == "iou":
        if getattr(cfg.optimization, "tapering", True): extras += "t"
        if getattr(cfg.optimization, "bending", True): extras += "b"
        if getattr(cfg.optimization, "reorient", True): extras += "r"
        if getattr(cfg.optimization, "pruning", False): extras += "p"
    folder_name = module_type + (f"_{extras}" if extras else "") + "_occ"
    output_dir = os.path.join("data", "output_npz", source_folder, folder_name)
    if small:
        output_dir = output_dir.replace("output_npz", "output_npz/small")
    os.makedirs(output_dir, exist_ok=True)

    # Save config to the output directory
    # cfg_path = os.path.join(output_dir, f"{split}_config.yaml")
    # with open(cfg_path, "w") as f:
    #     f.write(OmegaConf.to_yaml(cfg))

    output_npz = os.path.join(output_dir, f"{split}.npz")

    print(f"Loading {input_npz}...")
    print(f"Will save results to {output_npz}...")
    pred_handler = PredictionHandler.from_npz(input_npz)
    
    # Build dataloader from config (expects cfg.dataloader.*)
    dataloader = build_dataloader(cfg)
    assert dataloader.dataset.normalize == False, "Eval dataset should not be normalized."
    # Sanity-check: ensure prediction names align with dataset model_ids
    dataset_names = [m['model_id'] for m in dataloader.dataset.models]
    assert np.array_equal(pred_handler.names, dataset_names), (
        f"Object order differs between pred_handler and dataset."
    )
    valid_indices = list(range(len(pred_handler.names)))

    if small:
        valid_indices = random.sample(valid_indices, min(len(valid_indices), 128))
        subset = Subset(dataloader.dataset, valid_indices)
        dataloader = _build_dataloader(cfg, subset)
    print(f"Loaded {len(valid_indices)} objects from all categories out of {pred_handler.scale.shape[0]}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Store aggregated metrics
    aggregated_metrics = {
        'chamfer-L1': 0.0,
        'chamfer-L2': 0.0,
        'iou': 0.0,
        'f-score': 0.0,
        'f-score-15': 0.0,
        'f-score-20': 0.0,
        'num_primitives': 0.0,
        'count': 0
    }
    
    # Store per-object metrics for ranking
    object_metrics = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_indices = batch['idx']

        # Move tensors to device
        points = batch['abo_points'].to(device)
        normals = batch['abo_normals'].to(device)
        points_iou = batch['points_iou'].to(device)
        occupancies = batch['occupancies'].to(device)
        
        superq = BatchSuperQMulti(
            pred_handler=pred_handler,
            indices=batch_indices,
            device=device,
            cfg=cfg.optimization
        )
        
        param_groups = superq.get_param_groups()
        optimizer = torch.optim.Adam(param_groups)
        
        superq.update_handler(compute_meshes=False)
        
        # Cleanup to avoid OOM
        del superq
        del optimizer
        del param_groups
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Compute IoU using SDF
        with torch.no_grad():
            batch_ious = compute_ious_sdf_from_handler(pred_handler, batch_indices, points_iou, occupancies, device=device)

        # Evaluate mesh
        for b_idx in range(len(batch_indices)):
            idx = batch_indices[b_idx]
            try:
                mesh = pred_handler.get_mesh(idx, resolution=100, colors=False)
            except Exception as e:
                print(f"Error generating mesh for object {idx}: {e}")

            try:
                gt_pc = points[b_idx].cpu().numpy()
                gt_normal = normals[b_idx].cpu().numpy()
                out_dict_cur = eval_mesh(mesh, gt_pc, gt_normal, None, None)
                out_dict_cur['iou'] = float(batch_ious[b_idx]) if batch_ious is not None else 0.0
            except Exception as e:
                print(f"Eval mesh failed: {e}")
            
            num_prim = (pred_handler.exist[idx] > 0.5).sum()
            aggregated_metrics['chamfer-L1'] += out_dict_cur['chamfer-L1']
            aggregated_metrics['chamfer-L2'] += out_dict_cur['chamfer-L2']
            aggregated_metrics['iou'] += out_dict_cur.get('iou', 0.0)
            aggregated_metrics['f-score'] += out_dict_cur.get('f-score', 0.0)
            aggregated_metrics['f-score-15'] += out_dict_cur.get('f-score-15', 0.0)
            aggregated_metrics['f-score-20'] += out_dict_cur.get('f-score-20', 0.0)
            aggregated_metrics['num_primitives'] += num_prim
            aggregated_metrics['count'] += 1
            
            object_metrics.append({
                'index': idx.item(),
                'name': pred_handler.names[idx],
                'chamfer-L1': out_dict_cur['chamfer-L1'],
                'chamfer-L2': out_dict_cur['chamfer-L2'],
                'iou': out_dict_cur.get('iou', 0.0),
                'f-score': out_dict_cur.get('f-score', 0.0),
                'f-score-15': out_dict_cur.get('f-score-15', 0.0),
                'f-score-20': out_dict_cur.get('f-score-20', 0.0),
                'num_primitives': int(num_prim)
            })

    # Save results
    if module_type != "none" or small:
        print(f"Saving optimized results to {output_npz}...")
        if small:
            predictions = {
                'names': np.array(pred_handler.names)[valid_indices],
                'pc': np.array(pred_handler.pc)[valid_indices],
                'assign_matrix': np.array(pred_handler.assign_matrix)[valid_indices],
                'scale': np.array(pred_handler.scale)[valid_indices],
                'rotation': np.array(pred_handler.rotation)[valid_indices],
                'translation': np.array(pred_handler.translation)[valid_indices],
                'exponents': np.array(pred_handler.exponents)[valid_indices],
                'exist': np.array(pred_handler.exist)[valid_indices],
                'tapering': np.array(pred_handler.tapering)[valid_indices],
                'bending': np.array(pred_handler.bending)[valid_indices],
            }
            small_handler = PredictionHandler(predictions)
            small_handler.save_npz(output_npz)
        else:
            pred_handler.save_npz(output_npz)
    
    # Save detailed metrics
    metrics_csv = output_npz.replace(".npz", "_metrics.csv")
    print(f"Saving per-object metrics to {metrics_csv}...")
    import csv
    with open(metrics_csv, 'w', newline='') as csvfile:
        fieldnames = ['index', 'name', 'chamfer-L1', 'chamfer-L2', 'iou', 'f-score', 'f-score-15', 'f-score-20', 'num_primitives']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in object_metrics:
            writer.writerow(data)

    # Print Metrics
    count = aggregated_metrics['count']
    if count > 0:
        mean_chamfer_l1 = aggregated_metrics['chamfer-L1'] / count
        mean_chamfer_l2 = aggregated_metrics['chamfer-L2'] / count
        mean_iou = aggregated_metrics['iou'] / count
        mean_fscore = aggregated_metrics['f-score'] / count
        mean_fscore_15 = aggregated_metrics['f-score-15'] / count
        mean_fscore_20 = aggregated_metrics['f-score-20'] / count
        mean_num_primitives = aggregated_metrics['num_primitives'] / count
        
        print("\n----- Evaluation Results -----")
        print(f"{'mean_chamfer_l1':>25}: {mean_chamfer_l1:.6f}")
        print(f"{'mean_chamfer_l2':>25}: {mean_chamfer_l2:.6f}")
        print(f"{'mean_iou':>25}: {mean_iou:.6f}")
        print(f"{'mean_f-score':>25}: {mean_fscore:.6f}")
        print(f"{'mean_f-score-15':>25}: {mean_fscore_15:.6f}")
        print(f"{'mean_f-score-20':>25}: {mean_fscore_20:.6f}")
        print(f"{'avg_num_primitives':>25}: {mean_num_primitives:.6f}")
        
        # Sort by Chamfer-L1 descending (worst first)
        object_metrics.sort(key=lambda x: x['chamfer-L1'], reverse=True)
        print("\n----- Top 10 Worst Objects (by Chamfer-L1) -----")
        print(f"{'Index':<10} {'Name':<40} {'Chamfer-L1':<15}")
        for item in object_metrics[:10]:
            print(f"{item['index']:<10} {item['name']:<40} {item['chamfer-L1']:.6f}")
    else:
        print("No valid objects evaluated.")

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../configs", config_name="batch_eval_occ")
    def run_main(cfg: DictConfig):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        main(cfg)

    run_main()
