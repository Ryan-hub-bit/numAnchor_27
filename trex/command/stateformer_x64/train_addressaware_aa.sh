#!/usr/bin/env bash
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

cd /path/to/numAnchor/trex

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATA_BIN="${DATA_BIN:-/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/baseline_bin}"
AA_DATA_DIR="${AA_DATA_DIR:-/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/addressaware_aa_text}"
SAVE_DIR="${SAVE_DIR:-/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_20260422/addressaware_aa_pretrain_ckpt}"
LOG_PATH="${LOG_PATH:-/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_20260422/logs/addressaware_aa_pretrain.log}"

TOKENS_PER_SAMPLE="${TOKENS_PER_SAMPLE:-32}"
TOTAL_UPDATES="${TOTAL_UPDATES:-400}"
WARMUP_UPDATES="${WARMUP_UPDATES:-100}"
LR="${LR:-1e-4}"
MAX_SENTENCES="${MAX_SENTENCES:-512}"
BATCH_SIZE_VALID="${BATCH_SIZE_VALID:-512}"
VALIDATE_INTERVAL_UPDATES="${VALIDATE_INTERVAL_UPDATES:-200}"
SAVE_INTERVAL_UPDATES="${SAVE_INTERVAL_UPDATES:-200}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"

ENCODER_EMBED_DIM="${ENCODER_EMBED_DIM:-768}"
ENCODER_LAYERS="${ENCODER_LAYERS:-8}"
ENCODER_ATTENTION_HEADS="${ENCODER_ATTENTION_HEADS:-12}"

mkdir -p "${SAVE_DIR}" "$(dirname "${LOG_PATH}")"

python train.py \
  "${DATA_BIN}" \
  --task trex \
  --criterion trex \
  --arch trex \
  --sample-break-mode eos \
  --tokens-per-sample "${TOKENS_PER_SAMPLE}" \
  --optimizer adam \
  --adam-betas '(0.9,0.98)' \
  --adam-eps 1e-6 \
  --clip-norm 0.0 \
  --lr-scheduler polynomial_decay \
  --lr "${LR}" \
  --warmup-updates "${WARMUP_UPDATES}" \
  --total-num-update "${TOTAL_UPDATES}" \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --weight-decay 0.01 \
  --max-sentences "${MAX_SENTENCES}" \
  --max-update "${TOTAL_UPDATES}" \
  --log-format json \
  --log-interval "${LOG_INTERVAL}" \
  --validate-interval-updates "${VALIDATE_INTERVAL_UPDATES}" \
  --no-epoch-checkpoints \
  --save-interval-updates "${SAVE_INTERVAL_UPDATES}" \
  --keep-interval-updates 1 \
  --batch-size-valid "${BATCH_SIZE_VALID}" \
  --seed "${SEED}" \
  --num-workers "${NUM_WORKERS}" \
  --encoder-layers "${ENCODER_LAYERS}" \
  --encoder-embed-dim "${ENCODER_EMBED_DIM}" \
  --encoder-attention-heads "${ENCODER_ATTENTION_HEADS}" \
  --random-token-prob 0.5 \
  --mask-prob 0.2 \
  --save-dir "${SAVE_DIR}" \
  --memory-efficient-fp16 \
  --use-address-aware-embedding \
  --aa-data-dir "${AA_DATA_DIR}" | tee "${LOG_PATH}"
