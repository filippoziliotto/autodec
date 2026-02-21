import os
import sys
import torch
import numpy as np
import viser
import time
import argparse
from omegaconf import OmegaConf

from superdec.data.dataloader import ShapeNet, ABO
from superdec.utils.predictions_handler_extended import PredictionHandler
from superoptim.evaluation import get_outdict, eval_mesh, compute_ious_sdf

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
            batch_ious = compute_ious_sdf(handler, [0], item['points_iou'].unsqueeze(0), item['occupancies'].unsqueeze(0), device='cpu')
            print("IoU:", batch_ious)
        
        # Get mesh and add to scene
        mesh = handler.get_mesh(0, resolution=30)
        server.scene.add_mesh_trimesh("gt_superquadrics", mesh=mesh, visible=True)
    else:
        print("No GT superquadrics found in item.")

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset items")
    parser.add_argument("--dataset", type=str, default="abo", choices=["shapenet", "abo"], help="Dataset to use")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--index", type=int, default=0, help="Index of the item to visualize")
    args = parser.parse_args()

    cfg = OmegaConf.create({
        'shapenet': {
            'path': 'data/ShapeNet',
            'gt_train_path': 'data/output_npz/shapenet/shapenet_train_optimized_iou_bend.npz',
            # 'normalize': True,
            'load_occupancy': True,
            'use_fps': True,
        },
        'abo': {
            'path': 'data/ABO/processed-complete',
            'gt_train_path': 'data/output_npz/abo/abo_train_optimized_iou_bend.npz',
            # 'normalize': True,
            'load_occupancy': True,
            'use_fps': True,
        },
        'trainer': { 'augmentations': True }
    })

    # Initialize dataset
    if args.dataset == "shapenet":
        dataset = ShapeNet(args.split, cfg)
    elif args.dataset == "abo":
        dataset = ABO(args.split, cfg)
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
