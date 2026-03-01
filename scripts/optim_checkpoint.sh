CKPT_FOLDER=shapenet_iou_031
CKPT_FILE=epoch_1000.pt

python -m superdec.evaluate.to_npz --config-path="../../configs/optim_shapenet" \
  "dataloader.split=test" \
  "checkpoints_folder=checkpoints/$CKPT_FOLDER" \
  "checkpoint_file=$CKPT_FILE" \
  "output_dir=data/output_npz/$CKPT_FOLDER"

  # "optimization.reorient=true" \
python -m superoptim.batch_evaluate \
  "type=iou" \
  "optimization.tapering=true" \
  "optimization.bending=true" \
  "+source_folder=$CKPT_FOLDER"