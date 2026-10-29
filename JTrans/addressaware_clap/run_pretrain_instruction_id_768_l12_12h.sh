#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

NUMANCHOR_RUN_ROOT="${NUMANCHOR_RUN_ROOT:-/path/to/numanchor_runs}"
TRAIN_DIR="${TRAIN_DIR:-${NUMANCHOR_RUN_ROOT}/jtrans/addressaware_clap_instruction_id/train}"
OUT_ROOT="${OUT_ROOT:-${NUMANCHOR_RUN_ROOT}/jtrans/addressaware_clap_instruction_id_pretrain_large}"

VOCAB="${VOCAB:-${TRAIN_DIR}/vocab.pkl}"
TRAIN_CORPUS="${TRAIN_CORPUS:-${TRAIN_DIR}/pretrain_train_seq512.txt}"
VAL_CORPUS="${VAL_CORPUS:-${TRAIN_DIR}/pretrain_val_seq512.txt}"

SEQ_LEN="${SEQ_LEN:-512}"
SEGMENT_TYPES="${SEGMENT_TYPES:-513}"
MAX_TRAIN_FUNCTIONS="${MAX_TRAIN_FUNCTIONS:-110000}"
MAX_VAL_FUNCTIONS="${MAX_VAL_FUNCTIONS:-5000}"
HIDDEN="${HIDDEN:-768}"
LAYERS="${LAYERS:-12}"
HEADS="${HEADS:-12}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"

PRETRAIN_OUT="${OUT_ROOT}/pretrain_out_seq${SEQ_LEN}_h${HIDDEN}_l${LAYERS}_a${HEADS}_instrid_12h"
PRETRAIN_LOG_DIR="${OUT_ROOT}/logs_pretrain_seq${SEQ_LEN}_h${HIDDEN}_l${LAYERS}_a${HEADS}_instrid_12h"

mkdir -p "${PRETRAIN_OUT}" "${PRETRAIN_LOG_DIR}"

echo "[12h-pretrain] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[12h-pretrain] TRAIN_DIR=${TRAIN_DIR}"
echo "[12h-pretrain] OUT_ROOT=${OUT_ROOT}"
echo "[12h-pretrain] hidden=${HIDDEN} layers=${LAYERS} heads=${HEADS} batch=${BATCH_SIZE}"
echo "[12h-pretrain] max_train_functions=${MAX_TRAIN_FUNCTIONS} max_val_functions=${MAX_VAL_FUNCTIONS}"

PRETRAIN_ARGS=(
  --train_corpus "${TRAIN_CORPUS}"
  --val_corpus "${VAL_CORPUS}"
  --vocab "${VOCAB}"
  --output_dir "${PRETRAIN_OUT}"
  --log_dir "${PRETRAIN_LOG_DIR}"
  --seq_len "${SEQ_LEN}"
  --hidden "${HIDDEN}"
  --n_layers "${LAYERS}"
  --attn_heads "${HEADS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --device cuda
  --gradient_checkpointing
  --address_scale 2048
  --segment_types "${SEGMENT_TYPES}"
  --max_train_functions "${MAX_TRAIN_FUNCTIONS}"
  --max_val_functions "${MAX_VAL_FUNCTIONS}"
)

if [[ -f "${PRETRAIN_OUT}/latest_model.pt" ]]; then
  PRETRAIN_ARGS+=(--resume_checkpoint "${PRETRAIN_OUT}/latest_model.pt")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/pretrain.py" "${PRETRAIN_ARGS[@]}"

echo "[12h-pretrain] done"
echo "pretrain: ${PRETRAIN_OUT}"
