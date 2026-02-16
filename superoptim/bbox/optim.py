import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
from superdec.superdec import SuperDec
from superdec.utils.predictions_handler_extended import PredictionHandler
from superdec.data.dataloader import denormalize_outdict, denormalize_points
import open3d as o3d
import viser
import random
from superdec.data.dataloader import normalize_points, denormalize_outdict
from superdec.data.transform import rotate_around_axis
import time
from superdec.utils.visualizations import generate_ncolors
from tqdm import tqdm
import matplotlib.pyplot as plt

from .batch_superq import BatchSuperQMulti
from ..utils import plot_pred_handler

def visualize_handler(server, superq):
    # Expect batched tensors from BatchSuperQMulti; use batch 0
    pred_handler, meshes = superq.update_handler()
    mesh = meshes[superq.indices[0]]
    server.scene.add_mesh_trimesh("superquadrics", mesh=mesh, visible=True)


def main():
    if len(sys.argv) > 1:
        object_name = sys.argv[1]
    else:
        object_name = "round_table"
    pred_handler = PredictionHandler.from_npz(f"data/output_npz/objects/{object_name}.npz")
    print(f"Optimizing {pred_handler.names[0]}")
    
    superq = BatchSuperQMulti(
        pred_handler=pred_handler,
        indices=[0],
    )
    param_groups = superq.get_param_groups()
    optimizer = torch.optim.Adam(param_groups)

    pred_handler, meshes = superq.update_handler()
    orig_mesh = meshes[superq.indices[0]]

    server = viser.ViserServer()
    server.scene.set_up_direction([0.0, 1.0, 0.0])

    # Segmented pointcloud for batch 0
    points = pred_handler.pc[superq.indices[0]]
    assign_matrix = pred_handler.assign_matrix[superq.indices[0]]
    colors = generate_ncolors(assign_matrix.shape[1])
    segmentation = np.argmax(assign_matrix, axis=1)
    colored_pc = colors[segmentation]
    server.scene.add_point_cloud(
        name="/segmented_pointcloud",
        points=points,
        colors=colored_pc,
        point_size=0.005,
    )

    visualize_handler(server, superq)

    while True:
        time.sleep(10.0)

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()