#!/bin/bash

set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /path/to/numAnchor/trex

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_pretrain3q_10h_20260422/addressaware_aa_pretrain_ckpt/checkpoint_best.pt}"
DATA_DIR="${DATA_DIR:-/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/baseline_bin}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_addressaware_aa_funcsim_$(date -u +%Y%m%d_%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

python command/stateformer_x64/train_funcsim_retrieval.py \
  --mode addressaware_aa \
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --embedding-dim "${EMBEDDING_DIM:-256}" \
  --max-positions "${MAX_POSITIONS:-128}" \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --lr "${LR:-3e-5}" \
  --lr-backbone "${LR_BACKBONE:-3e-6}" \
  --temperature "${TEMPERATURE:-0.07}" \
  --valid-fraction "${VALID_FRACTION:-0.02}" \
  --train-positive-samples "${TRAIN_POSITIVE_SAMPLES:-50000}" \
  --valid-positive-samples "${VALID_POSITIVE_SAMPLES:-5000}" \
  --device "${DEVICE:-cuda}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --seed "${SEED:-42}"
