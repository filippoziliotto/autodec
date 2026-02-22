CONFIG_FOLDER=../configs/optim_shapenet

# python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "type=none"
# python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "type=base"
python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER"
python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.reorient=true"

python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.tapering=true"
python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.bending=true"
python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.tapering=true" "optimization.bending=true"
python -m superoptim.batch_evaluate --config-path="$CONFIG_FOLDER" "optimization.tapering=true" "optimization.bending=true" "optimization.reorient=true"