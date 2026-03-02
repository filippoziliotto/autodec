import os
import sys
import torch
import numpy as np
import viser
import time
import argparse
from omegaconf import OmegaConf

from superdec.data.dataloader import ShapeNet, ABO, ASE
from superdec.utils.predictions_handler_extended import PredictionHandler
from superoptim.evaluation import get_outdict, eval_mesh, compute_ious_sdf_from_handler

def visualize_item(server, item):
    # Clear previous items
    server.scene.reset()
    server.scene.set_up_direction([0.0, 1.0, 0.0])

    # Visualize point cloud
    points = item['points'].numpy()
    colors = np.tile(np.array([0.5, 0.5, 0.5]), (points.shape[0], 1))
    server.scene.add_point_cloud(
        name="/pointcloud",
        points=points,
        colors=colors,
        point_size=0.01,
    )

    if 'abo_points' in item:
        abo_points = item['abo_points'].numpy()
        colors = np.tile(np.array([1, 0.5, 0.5]), (abo_points.shape[0], 1))
        server.scene.add_point_cloud(
            name="/pointcloud_abo",
            points=abo_points,
            colors=colors,
            point_size=0.01,
        )

    if 'points_iou' in item:
        iou_occ_points = np.array(item['points_iou'][item['occupancies'] > 0])
        green_colors = np.tile(np.array([0.0, 1.0, 0.0]), (iou_occ_points.shape[0], 1))
        server.scene.add_point_cloud(
            name="/iou_occupied_points",
            points=iou_occ_points,
            colors=green_colors,
            point_size=0.005,
            visible=True,
        )

    # Visualize GT superquadrics if available
    if 'gt_scale' in item:
        print("Visualizing GT superquadrics...")
        # Create a dummy PredictionHandler to generate meshes
        num_primitives = item['gt_scale'].shape[0]
        
        # Convert tensors to numpy arrays
        scale = item['gt_scale'].numpy()
        shape = item['gt_shape'].numpy()
        trans = item['gt_trans'].numpy()
        rotate = item['gt_rotate'].numpy()
        exist = item['gt_exist'].numpy()
        tapering = item['gt_tapering'].numpy()
        bending = item['gt_bending'].numpy()

        # Create dummy handler
        handler = PredictionHandler({
            'assign_matrix': np.zeros((1, 16, 4096)),
            'pc':np.expand_dims(points, 0),
            'scale':np.expand_dims(scale, 0),
            'exponents':np.expand_dims(shape, 0),
            'translation':np.expand_dims(trans, 0),
            'rotation':np.expand_dims(rotate, 0),
            'exist':np.expand_dims(exist, 0),
            'tapering':np.expand_dims(tapering, 0),
            'bending':np.expand_dims(bending, 0),
            'names':[item['model_id']]
        })

        if 'points_iou' in item:
            batch_ious = compute_ious_sdf_from_handler(handler, [0], item['points_iou'].unsqueeze(0), item['occupancies'].unsqueeze(0), device='cpu')
            print("IoU:", batch_ious)

        if 'gt_sq_points' in item:
            pts = item['gt_sq_points'][(exist>0.5).squeeze(1)].numpy().reshape(-1, 3)
            red_colors = np.tile(np.array([1.0, 0.0, 0.0]), (pts.shape[0], 1))
            server.scene.add_point_cloud(
                name="/geom_points",
                points=pts,
                colors=red_colors,
                point_size=0.005,
                visible=True,
            )
            
        
        # Get mesh and add to scene
        mesh = handler.get_mesh(0, resolution=30)
        server.scene.add_mesh_trimesh("gt_superquadrics", mesh=mesh, visible=True)
    else:
        print("No GT superquadrics found in item.")

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset items")
    parser.add_argument("--dataset", type=str, default="ase", choices=["shapenet", "abo", "ase"], help="Dataset to use")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--index", type=int, default=0, help="Index of the item to visualize")
    args = parser.parse_args()

    cfg = OmegaConf.create({
        'shapenet': {
            'path': 'data/ShapeNet',
            'categories': None,
            'gt_train_path': 'data/output_npz/shapenet/shapenet_train_optimized_iou_bend.npz',
            'normalize': True,
            'load_occupancy': True,
            # 'use_fps': True,
        },
        'abo': {
            'path': 'data/ABO/processed-complete',
            'gt_train_path': 'data/output_npz/abo/abo_train_optimized_iou_bend.npz',
            'normalize': True,
            'load_occupancy': True,
            'geometric_samples': 256,
            # 'use_fps': True,
        },
        'ase': {
            'instances_path': 'data/ase/scenes',
            'csv_path': 'data/ase/scenes/object_info_new.csv',
            'abo_path': 'data/ABO/processed-complete',
            'load_occupancy': True,
            'normalize': True,
        },
        'trainer': { 
            'augmentations': True,
            'occlusions': True
        }
    })

    # Initialize dataset
    if args.dataset == "shapenet":
        dataset = ShapeNet(args.split, cfg)
    elif args.dataset == "abo":
        dataset = ABO(args.split, cfg)
    elif args.dataset == "ase":
        dataset = ASE(args.split, cfg)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Loaded {args.dataset} dataset ({args.split} split) with {len(dataset)} items.")

    if args.index >= len(dataset):
        print(f"Error: Index {args.index} is out of bounds for dataset of size {len(dataset)}.")
        return

    # Get item
    item = dataset[args.index]
    print(f"Visualizing item {args.index} (Model ID: {item['model_id']})")

    # Start Viser server
    server = viser.ViserServer()
    print("Viser server started. Open http://localhost:8080 in your browser.")

    # Visualize
    visualize_item(server, item)

    # Keep server running
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
