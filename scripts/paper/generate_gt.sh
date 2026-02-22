CONFIG_FOLDER=configs/optim_shapenet

# python -m superdec.evaluate.to_npz --config-path="../../$CONFIG_FOLDER"
python -m superoptim.batch_evaluate --config-path="../$CONFIG_FOLDER" --config-name="batch_optim"

# python -m superdec.evaluate.to_npz --config-path="../../$CONFIG_FOLDER" "dataloader.split=val"
python -m superoptim.batch_evaluate --config-path="../$CONFIG_FOLDER" --config-name="batch_optim" "prefix=shapenet/shapenet_val" "shapenet.split=val"