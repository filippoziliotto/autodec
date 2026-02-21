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
        handler = PredictionHandler(
            scale=np.expand_dict(scale, 0),
            shape=np.expand_dict(shape, 0),
            trans=np.expand_dict(trans, 0),
            rotate=np.expand_dict(rotate, 0),
            exist=np.expand_dict(exist, 0),
            tapering=np.expand_dict(tapering, 0),
            bending=np.expand_dict(bending, 0),
            names=[item['model_id']]
        )

        # Get mesh and add to scene
        mesh = handler.get_mesh(0, resolution=30)
        server.scene.add_mesh_trimesh("gt_superquadrics", mesh=mesh, visible=True)
    else:
        print("No GT superquadrics found in item.")

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset items")
    parser.add_argument("--dataset", type=str, default="shapenet", choices=["shapenet", "abo"], help="Dataset to use")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--index", type=int, default=0, help="Index of the item to visualize")
    parser.add_argument("--config", type=str, default="configs/mamba.toml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

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
