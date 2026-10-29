#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

NUMANCHOR_RUN_ROOT="${NUMANCHOR_RUN_ROOT:-/path/to/numanchor_runs}"
TRAIN_DIR="${TRAIN_DIR:-${NUMANCHOR_RUN_ROOT}/jtrans/addressaware_clap_instruction_id/train}"
OUT_ROOT="${OUT_ROOT:-${NUMANCHOR_RUN_ROOT}/jtrans/addressaware_clap_instruction_id_pretrain}"

SEQ_LEN="${SEQ_LEN:-512}"
SEGMENT_TYPES="${SEGMENT_TYPES:-513}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-6}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TRAIN_FUNCTIONS="${MAX_TRAIN_FUNCTIONS:-}"
MAX_VAL_FUNCTIONS="${MAX_VAL_FUNCTIONS:-}"

VOCAB="${VOCAB:-${TRAIN_DIR}/vocab.pkl}"
TRAIN_CORPUS="${TRAIN_CORPUS:-${TRAIN_DIR}/pretrain_train_seq512.txt}"
VAL_CORPUS="${VAL_CORPUS:-${TRAIN_DIR}/pretrain_val_seq512.txt}"

PRETRAIN_OUT="${OUT_ROOT}/pretrain_out_seq${SEQ_LEN}_h384_l6_addrtype_scale2048_instrid"
PRETRAIN_LOG_DIR="${OUT_ROOT}/logs_pretrain_seq${SEQ_LEN}_h384_l6_addrtype_scale2048_instrid"

mkdir -p "${PRETRAIN_OUT}" "${PRETRAIN_LOG_DIR}"

echo "[instruction-id AAE pretrain-only] ROOT=${SCRIPT_DIR}"
echo "[instruction-id AAE pretrain-only] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[instruction-id AAE pretrain-only] OUT_ROOT=${OUT_ROOT}"
echo "[instruction-id AAE pretrain-only] segment_types=${SEGMENT_TYPES}"
echo "[instruction-id AAE pretrain-only] max_train_functions=${MAX_TRAIN_FUNCTIONS:-all}"
echo "[instruction-id AAE pretrain-only] max_val_functions=${MAX_VAL_FUNCTIONS:-all}"
echo "[instruction-id AAE pretrain-only] num_workers=${NUM_WORKERS}"

PRETRAIN_ARGS=(
  --train_corpus "${TRAIN_CORPUS}"
  --val_corpus "${VAL_CORPUS}"
  --vocab "${VOCAB}"
  --output_dir "${PRETRAIN_OUT}"
  --log_dir "${PRETRAIN_LOG_DIR}"
  --seq_len "${SEQ_LEN}"
  --hidden 384
  --n_layers 6
  --attn_heads 8
  --batch_size "${PRETRAIN_BATCH_SIZE}"
  --epochs "${PRETRAIN_EPOCHS}"
  --lr "${PRETRAIN_LR}"
  --device cuda
  --gradient_checkpointing
  --address_scale 2048
  --segment_types "${SEGMENT_TYPES}"
  --num_workers "${NUM_WORKERS}"
)

if [[ -f "${PRETRAIN_OUT}/latest_model.pt" ]]; then
  PRETRAIN_ARGS+=(--resume_checkpoint "${PRETRAIN_OUT}/latest_model.pt")
fi
if [[ -n "${MAX_TRAIN_FUNCTIONS}" ]]; then
  PRETRAIN_ARGS+=(--max_train_functions "${MAX_TRAIN_FUNCTIONS}")
fi
if [[ -n "${MAX_VAL_FUNCTIONS}" ]]; then
  PRETRAIN_ARGS+=(--max_val_functions "${MAX_VAL_FUNCTIONS}")
fi

echo "[1/1] pretrain"
"${PYTHON_BIN}" "${SCRIPT_DIR}/pretrain.py" "${PRETRAIN_ARGS[@]}"

echo "[instruction-id AAE pretrain-only] done"
echo "pretrain: ${PRETRAIN_OUT}"
