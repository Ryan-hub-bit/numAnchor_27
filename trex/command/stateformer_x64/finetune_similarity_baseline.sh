#!/usr/bin/env bash
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

cd /path/to/numAnchor/trex

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ROOT="${RUN_ROOT:-/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_baseline_finetune_20260422}"
BIN_DIR="${BIN_DIR:-${RUN_ROOT}/baseline_similarity_bin}"
SAVE_DIR="${SAVE_DIR:-${RUN_ROOT}/baseline_similarity_ckpt}"
LOG_PATH="${LOG_PATH:-${RUN_ROOT}/logs/baseline_similarity_train.log}"
RESTORE_FILE="${RESTORE_FILE:-/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_20260422/baseline_pretrain_ckpt/checkpoint_best.pt}"

TOTAL_UPDATES="${TOTAL_UPDATES:-6000}"
WARMUP_UPDATES="${WARMUP_UPDATES:-100}"
LR="${LR:-5e-5}"
MAX_POSITIONS="${MAX_POSITIONS:-128}"
MAX_SENTENCES="${MAX_SENTENCES:-64}"
MAX_SENTENCES_VALID="${MAX_SENTENCES_VALID:-128}"
UPDATE_FREQ="${UPDATE_FREQ:-4}"
VALIDATE_INTERVAL_UPDATES="${VALIDATE_INTERVAL_UPDATES:-500}"
SAVE_INTERVAL_UPDATES="${SAVE_INTERVAL_UPDATES:-500}"
SEED="${SEED:-42}"

ENCODER_EMBED_DIM="${ENCODER_EMBED_DIM:-768}"
ENCODER_LAYERS="${ENCODER_LAYERS:-8}"
ENCODER_ATTENTION_HEADS="${ENCODER_ATTENTION_HEADS:-12}"

mkdir -p "${SAVE_DIR}" "$(dirname "${LOG_PATH}")"

python train.py "${BIN_DIR}" \
  --task similarity \
  --criterion similarity \
  --arch trex \
  --max-positions "${MAX_POSITIONS}" \
  --truncate-sequence \
  --max-sentences "${MAX_SENTENCES}" \
  --max-sentences-valid "${MAX_SENTENCES_VALID}" \
  --update-freq "${UPDATE_FREQ}" \
  --reset-optimizer --reset-dataloader --reset-meters \
  --required-batch-size-multiple 1 \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-06 \
  --clip-norm 0.0 \
  --lr-scheduler polynomial_decay \
  --lr "${LR}" \
  --warmup-updates "${WARMUP_UPDATES}" \
  --total-num-update "${TOTAL_UPDATES}" \
  --max-update "${TOTAL_UPDATES}" \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --weight-decay 0.01 \
  --best-checkpoint-metric AUC_pair \
  --maximize-best-checkpoint-metric \
  --find-unused-parameters \
  --no-epoch-checkpoints \
  --validate-interval-updates "${VALIDATE_INTERVAL_UPDATES}" \
  --save-interval-updates "${SAVE_INTERVAL_UPDATES}" \
  --keep-interval-updates 1 \
  --log-format json \
  --log-interval 20 \
  --seed "${SEED}" \
  --num-workers 0 \
  --encoder-layers "${ENCODER_LAYERS}" \
  --encoder-embed-dim "${ENCODER_EMBED_DIM}" \
  --encoder-attention-heads "${ENCODER_ATTENTION_HEADS}" \
  --save-dir "${SAVE_DIR}" \
  --restore-file "${RESTORE_FILE}" \
  --memory-efficient-fp16 | tee "${LOG_PATH}"
