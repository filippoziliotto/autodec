#!/bin/sh

#SBATCH --job-name=SuperOptim
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out

DATASET_DIR="abo"

python -m superoptim.batch_evaluate --type none --prefix ${DATASET_DIR}/abo_train
python -m superoptim.batch_evaluate --type empty --prefix ${DATASET_DIR}/abo_train
python -m superoptim.batch_evaluate --type iou --prefix ${DATASET_DIR}/abo_train
python -m superoptim.batch_evaluate --type iou_bend --prefix ${DATASET_DIR}/abo_train
python -m superoptim.batch_evaluate --type bbox --prefix ${DATASET_DIR}/abo_train

python superoptim/select_best.py \
  data/output_npz/${DATASET_DIR}/abo_train_optimized_empty_metrics.csv \
  data/output_npz/${DATASET_DIR}/abo_train_optimized_iou_metrics.csv \
  data/output_npz/${DATASET_DIR}/abo_train_optimized_iou_bend_metrics.csv \
  data/output_npz/${DATASET_DIR}/abo_train_optimized_bbox_metrics.csv \
  --save-npz data/output_npz/${DATASET_DIR}/abo_train_optimized_combined.npz