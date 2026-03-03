#!/bin/bash
set -euo pipefail  # stop on error, undefined variable, or pipeline failure

# trap to print the failing command
trap 'echo "Error occurred at command: $BASH_COMMAND"' ERR


OUTPUT_NPZ_DIR=data/output_replica/ # path to the folder where to save the output .npz files
SCENE_NAME=room0 # name of the scene (used to name the output .npz file)
OBJECTS_SCENE_DIR="data/output_replica/${SCENE_NAME}/ply"  # path to the folder containing the .ply files of all the segmented objects in the scene
Z_UP=true
mkdir -p $OBJECTS_SCENE_DIR
cp /capstor/store/cscs/swissai/a115/data/replica_mask_3d_numpy/ply/${SCENE_NAME}* $OBJECTS_SCENE_DIR

python superdec/utils/ply_to_npz.py --input_path="$OBJECTS_SCENE_DIR" --scene_name="$SCENE_NAME"

python superdec/evaluate/to_npz.py checkpoint_file="epoch_1000.pt" checkpoints_folder="checkpoints/shapenet_geom_022" output_dir="$OUTPUT_NPZ_DIR" dataset=scene scene.name="$SCENE_NAME" scene.z_up="$Z_UP"

# python superdec/visualization/object_visualizer.py dataset=scene split="$SCENE_NAME" npz_folder="$OUTPUT_NPZ_DIR"