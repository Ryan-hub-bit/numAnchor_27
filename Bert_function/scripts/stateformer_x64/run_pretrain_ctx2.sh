#!/bin/bash

set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree

TRAIN_CFG="${TRAIN_CFG:-/path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/train_cfg.txt}"
VAL_CFG="${VAL_CFG:-/path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/val_cfg.txt}"
VOCAB="${VOCAB:-/path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/vocab/vocab_stateformer_x64_ctx2.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/numAnchor/output/palmtree_output/pretrain/stateformer_x64_ctx2}"
LOG_DIR="${LOG_DIR:-/path/to/numAnchor_artifact/log/Palmtree/pretrain_stateformer_x64_ctx2}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd /path/to/numAnchor/PalmTree

train_args=(
  --cfg_train "${TRAIN_CFG}" \
  --cfg_val "${VAL_CFG}" \
  --vocab "${VOCAB}" \
  --hidden "${HIDDEN:-768}" \
  --layers "${LAYERS:-12}" \
  --attn_heads "${ATTN_HEADS:-12}" \
  --seq_len "${SEQ_LEN:-20}" \
  --epochs "${EPOCHS:-12}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --lr "${LR:-1e-4}" \
  --warmup_steps "${WARMUP_STEPS:-1000}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --data_percentage "${DATA_PERCENTAGE:-1.0}" \
  --val_percentage "${VAL_PERCENTAGE:-1.0}" \
  --token_mask_prob 0.15 \
  --instruction_mask_prob 0.15 \
  --enable_imc \
  --enable_mlm \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --log_freq "${LOG_FREQ:-1000}"
)

if [ "${USE_CUDA:-true}" = "true" ]; then
  train_args+=(--cuda)
fi

if [ "${MULTI_GPU:-false}" = "true" ]; then
  train_args+=(--multi_gpu)
fi

if [ "${USE_AMP:-false}" = "true" ]; then
  train_args+=(--amp)
fi

if [ "${GRADIENT_CHECKPOINTING:-false}" = "true" ]; then
  train_args+=(--gradient_checkpointing)
fi

python train.py "${train_args[@]}"
