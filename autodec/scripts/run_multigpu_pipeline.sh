#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/common.sh"
ensure_fast_sampler

NUM_GPUS="${NUM_GPUS:-}"
if [[ -z "${NUM_GPUS}" ]]; then
  NUM_GPUS="$(python - <<'PY'
import torch

print(torch.cuda.device_count())
PY
)"
fi
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "NUM_GPUS must be at least 1; detected ${NUM_GPUS}" >&2
  exit 1
fi

PHASE1_EPOCHS="${PHASE1_EPOCHS:-100}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-200}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
PHASE1_RUN_NAME="${PHASE1_RUN_NAME:-autodec_phase1_ddp${NUM_GPUS}_100ep_bs8}"
PHASE2_RUN_NAME="${PHASE2_RUN_NAME:-autodec_phase2_ddp${NUM_GPUS}_200ep_bs8}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-autodec_test_eval_ddp${NUM_GPUS}_phase2_200ep_bs8}"

PHASE1_CKPT="${PHASE1_CKPT:-${CHECKPOINT_ROOT}/${PHASE1_RUN_NAME}/epoch_${PHASE1_EPOCHS}.pt}"
PHASE2_CKPT="${PHASE2_CKPT:-${CHECKPOINT_ROOT}/${PHASE2_RUN_NAME}/epoch_${PHASE2_EPOCHS}.pt}"

echo "--- AutoDec multi-GPU phase 1"
echo "--- GPUs: ${NUM_GPUS}; epochs: ${PHASE1_EPOCHS}; batch size per GPU: ${BATCH_SIZE_PER_GPU}"
torchrun --nproc_per_node="${NUM_GPUS}" -m autodec.training.train \
  --config-name train_phase1 \
  run_name="${PHASE1_RUN_NAME}" \
  trainer.num_epochs="${PHASE1_EPOCHS}" \
  trainer.batch_size="${BATCH_SIZE_PER_GPU}" \
  "$@"

echo "--- AutoDec multi-GPU phase 2"
echo "--- GPUs: ${NUM_GPUS}; epochs: ${PHASE2_EPOCHS}; batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "--- resuming from ${PHASE1_CKPT}"
torchrun --nproc_per_node="${NUM_GPUS}" -m autodec.training.train \
  --config-name train_phase2 \
  run_name="${PHASE2_RUN_NAME}" \
  checkpoints.resume_from="${PHASE1_CKPT}" \
  trainer.num_epochs="${PHASE2_EPOCHS}" \
  trainer.batch_size="${BATCH_SIZE_PER_GPU}" \
  "$@"

echo "--- AutoDec test evaluation"
echo "--- evaluating ${PHASE2_CKPT}"
python -m autodec.eval.run \
  --config-name eval_test \
  run_name="${EVAL_RUN_NAME}" \
  checkpoints.resume_from="${PHASE2_CKPT}" \
  "$@"

echo "--- AutoDec multi-GPU pipeline finished"
