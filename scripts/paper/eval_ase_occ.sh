declare -A CKPTS=(
  [shapenet_geom_022]=epoch_1000.pt
  [shapenet_geom_exist_0.01_f_lolr_big]=epoch_1000.pt
  [shapenet_geom_ase_004_lowfilter_hq]=epoch_100.pt
  [shapenet_geom_ase_004_lowfilter_hq2]=epoch_100.pt
  [shapenet_geom_ase_finetune]=epoch_100.pt
  [shapenet_iou_371]=epoch_1000.pt
  [normalized]=ckpt.pt
)

# for CKPT_FOLDER in shapenet_geom_022 shapenet_geom_exist_0.01_f_lolr_big \
#                    shapenet_geom_ase_004_lowfilter_hq shapenet_geom_ase_004_lowfilter_hq2 \
#                    shapenet_geom_ase_finetune shapenet_iou_371 normalized; do
for CKPT_FOLDER in shapenet_geom_ase_finetune; do
  CKPT_FILE=${CKPTS[$CKPT_FOLDER]}

  python -m superdec.evaluate.to_npz --config-name="save_npz_occ" \
    "checkpoints_folder=checkpoints/$CKPT_FOLDER" \
    "checkpoint_file=$CKPT_FILE" \
    "output_dir=data/output_npz/$CKPT_FOLDER"

  python -m superoptim.batch_evaluate_occ \
    "source_folder=$CKPT_FOLDER"
done
