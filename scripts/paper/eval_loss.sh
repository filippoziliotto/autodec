CONFIG_FOLDER=../configs/optim_shapenet

# python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "+optimization.w_sdf=0.0" "+optimization.w_bbox=0.0" "+optimization.w_overlap=0.0"
# python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "+optimization.w_sdf=0.0" "+optimization.w_bbox=0.0"
# python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "+optimization.w_sdf=0.0"
# python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" 

python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.tapering=true" "optimization.bending=true" "+optimization.w_sdf=0.0" "+optimization.w_bbox=0.0" "+optimization.w_overlap=0.0"
python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.tapering=true" "optimization.bending=true" "+optimization.w_sdf=0.0" "+optimization.w_bbox=0.0"
python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.tapering=true" "optimization.bending=true" "+optimization.w_sdf=0.0"
python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.tapering=true" "optimization.bending=true"