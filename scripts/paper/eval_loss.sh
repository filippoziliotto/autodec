CONFIG_FOLDER=../configs/optim_shapenet

# python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "+optimization.w_sdf=0.0" "+optimization.w_bbox=0.0" "+optimization.w_overlap=0.0"
# python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "+optimization.w_sdf=0.0" "+optimization.w_bbox=0.0"
# python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "+optimization.w_sdf=0.0"
# python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" 

python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.tapering=true" "optimization.bending=true" "+optimization.w_overlap=0.0" "+optimization.w_bbox=0.0" "+optimization.w_sdf=0.0"
python -m  superoptim.compute_overlap data/output_npz/shapenet/iou_tb/test.npz 
mv data/output_npz/shapenet/iou_tb data/output_npz/shapenet/ablation/o0b0s0
python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.tapering=true" "optimization.bending=true" "+optimization.w_overlap=0.0" "+optimization.w_bbox=0.0"
python -m  superoptim.compute_overlap data/output_npz/shapenet/iou_tb/test.npz 
mv data/output_npz/shapenet/iou_tb data/output_npz/shapenet/ablation/o0b0
# python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.tapering=true" "optimization.bending=true" "+optimization.w_overlap=0.0"
python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.tapering=true" "optimization.bending=true"
python -m  superoptim.compute_overlap data/output_npz/shapenet/iou_tb/test.npz 