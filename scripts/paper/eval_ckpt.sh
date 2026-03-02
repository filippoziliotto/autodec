CONFIG_FOLDER=configs/optim_shapenet
CKPT_FOLDER=shapenet_iou_371
CKPT_FILE=epoch_1000.pt

python -m superdec.evaluate.to_npz --config-path="../../$CONFIG_FOLDER" \
  "dataloader.split=test" \
  "checkpoints_folder=checkpoints/$CKPT_FOLDER" \
  "checkpoint_file=$CKPT_FILE" \
  "output_dir=data/output_npz/$CKPT_FOLDER"

python -m superoptim.batch_evaluate --config-path="../$CONFIG_FOLDER" \
  --config-name="batch_optim" \
  "type=none" \
  "shapenet.split=test" \
  "+source_folder=$CKPT_FOLDER"

python -m superoptim.batch_evaluate --config-path="../$CONFIG_FOLDER" \
  --config-name="batch_optim" \
  "shapenet.split=test" \
  "+source_folder=$CKPT_FOLDER"