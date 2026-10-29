#!/bin/bash

# Conservative HPC pretraining launcher for the StateFormer x86-64 ctx=2 corpus.
# Design goals:
# 1. Slightly reduce corpus size without changing the training objective
# 2. Avoid DDP RAM blow-up because train.py loads the whole corpus into memory
# 3. Use 4x A100 via DataParallel + AMP to finish epochs fast enough to get
#    several checkpoints within a 3-day walltime

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "PalmTree StateFormer ctx=2 (HPC)"
echo "========================================"
echo "Node:                 $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID:         ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_GPUS:           ${SLURM_GPUS:-<unset>}"
echo ""

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/numAnchor/PalmTree}"
DATA_ROOT="${DATA_ROOT:-/work/kliu14/Palmtreedata/stateformer_x64_dataset/pretrain_ctx2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/work/kliu14/Palmtreeoutput}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-stateformer_x64_ctx2_conservative}"

TRAIN_FULL="${TRAIN_FULL:-${DATA_ROOT}/splits/train_cfg.txt}"
VAL_CFG="${VAL_CFG:-${DATA_ROOT}/splits/val_cfg.txt}"
VOCAB_PATH="${VOCAB_PATH:-${DATA_ROOT}/vocab/vocab_stateformer_x64_ctx2.pkl}"

# Conservative reduction:
# keep validation file unchanged, and sample only the training file to 70%.
TRAIN_SAMPLE_RATIO="${TRAIN_SAMPLE_RATIO:-0.70}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_CFG="${TRAIN_CFG:-${DATA_ROOT}/splits/train_cfg_sample70_seed42.txt}"

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/log/${EXPERIMENT_NAME}}"

# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------
TASKS="${TASKS:---enable_imc --enable_mlm}"
USE_ADDRESS_EMBEDDING="${USE_ADDRESS_EMBEDDING:-true}"
USE_VAR_EMBEDDING="${USE_VAR_EMBEDDING:-true}"

HIDDEN="${HIDDEN:-768}"
LAYERS="${LAYERS:-12}"
ATTN_HEADS="${ATTN_HEADS:-12}"
SEQ_LEN="${SEQ_LEN:-20}"

# DataParallel uses one Python process, so batch_size is the global batch.
# 512 is a steadier default on 4xA100 80G for this corpus size.
BATCH_SIZE="${BATCH_SIZE:-512}"
# Keep workers low because train.py loads the corpus into memory and extra
# workers/pin-memory threads increase host RAM pressure.
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-10}"
WARMUP="${WARMUP:-1000}"
EARLY_STOPPING="${EARLY_STOPPING:-10}"
LOG_FREQ="${LOG_FREQ:-500}"

TOKEN_MASK_PROB="${TOKEN_MASK_PROB:-0.15}"
INSTRUCTION_MASK_PROB="${INSTRUCTION_MASK_PROB:-0.15}"
VAL_PERCENTAGE="${VAL_PERCENTAGE:-0.10}"

# We intentionally avoid DDP here because train.py loads the entire corpus into
# memory. DDP would replicate the 10GB+ sampled corpus 4 times and risks RAM OOM.
USE_AMP="${USE_AMP:-true}"
USE_MULTI_GPU="${USE_MULTI_GPU:-true}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"

if [ "${USE_ADDRESS_EMBEDDING}" = "true" ]; then
  ADDRESS_FLAG=""
else
  ADDRESS_FLAG="--disable_address_embedding"
fi

if [ "${USE_VAR_EMBEDDING}" = "true" ]; then
  VAR_FLAG=""
else
  VAR_FLAG="--disable_var_embedding"
fi

echo "Configuration:"
echo "  Project root:         ${PROJECT_ROOT}"
echo "  Data root:            ${DATA_ROOT}"
echo "  Full train:           ${TRAIN_FULL}"
echo "  Sampled train:        ${TRAIN_CFG}"
echo "  Val:                  ${VAL_CFG}"
echo "  Vocab:                ${VOCAB_PATH}"
echo "  Output:               ${OUTPUT_DIR}"
echo "  Log dir:              ${LOG_DIR}"
echo "  Tasks:                ${TASKS}"
echo "  Sample ratio:         ${TRAIN_SAMPLE_RATIO}"
echo "  Batch size:           ${BATCH_SIZE}"
echo "  Seq len:              ${SEQ_LEN}"
echo "  Epochs:               ${EPOCHS}"
echo "  Learning rate:        ${LR}"
echo "  Workers:              ${NUM_WORKERS}"
echo "  AMP:                  ${USE_AMP}"
echo "  Multi-GPU:            ${USE_MULTI_GPU}"
echo "  Gradient checkpoint:  ${USE_GRADIENT_CHECKPOINTING}"
echo ""

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "$(dirname "${TRAIN_CFG}")"

for path in "${PROJECT_ROOT}" "${TRAIN_FULL}" "${VAL_CFG}" "${VOCAB_PATH}"; do
  if [ ! -e "${path}" ]; then
    echo -e "${YELLOW}Missing required path: ${path}${NC}"
    exit 1
  fi
done

# -----------------------------------------------------------------------------
# Build conservative sampled training corpus if it does not already exist
# -----------------------------------------------------------------------------
if [ ! -f "${TRAIN_CFG}" ]; then
  echo "Creating conservative sampled training corpus..."
  python3 - <<PY
import random
from pathlib import Path

src = Path("${TRAIN_FULL}")
dst = Path("${TRAIN_CFG}")
ratio = float("${TRAIN_SAMPLE_RATIO}")
seed = int("${TRAIN_SEED}")
rng = random.Random(seed)

kept = 0
seen = 0
with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        seen += 1
        if rng.random() < ratio:
            fout.write(line)
            kept += 1

print(f"Sampled train corpus created: {dst}")
print(f"Input lines kept: {kept}/{seen} ({kept / max(seen, 1):.4%})")
PY
else
  echo "Sampled training corpus already exists: ${TRAIN_CFG}"
fi

cd "${PROJECT_ROOT}"

TRAIN_CMD=(
  python train.py
  --cfg_train "${TRAIN_CFG}"
  --cfg_val "${VAL_CFG}"
  --vocab "${VOCAB_PATH}"
  --hidden "${HIDDEN}"
  --layers "${LAYERS}"
  --attn_heads "${ATTN_HEADS}"
  --seq_len "${SEQ_LEN}"
  --dropout 0.1
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --warmup_steps "${WARMUP}"
  --num_workers "${NUM_WORKERS}"
  --early_stopping_patience "${EARLY_STOPPING}"
  --token_mask_prob "${TOKEN_MASK_PROB}"
  --instruction_mask_prob "${INSTRUCTION_MASK_PROB}"
  --data_percentage 1.0
  --val_percentage "${VAL_PERCENTAGE}"
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
  --log_freq "${LOG_FREQ}"
  --resume
  --cuda
)

if [ "${USE_MULTI_GPU}" = "true" ]; then
  TRAIN_CMD+=(--multi_gpu)
fi

if [ "${USE_AMP}" = "true" ]; then
  TRAIN_CMD+=(--amp)
fi

if [ "${USE_GRADIENT_CHECKPOINTING}" = "true" ]; then
  TRAIN_CMD+=(--gradient_checkpointing)
fi

if [ -n "${ADDRESS_FLAG}" ]; then
  TRAIN_CMD+=(${ADDRESS_FLAG})
fi

if [ -n "${VAR_FLAG}" ]; then
  TRAIN_CMD+=(${VAR_FLAG})
fi

# TASKS is expected to be a small flat string like "--enable_imc --enable_mlm"
# shellcheck disable=SC2206
TASK_ARGS=(${TASKS})
TRAIN_CMD+=("${TASK_ARGS[@]}")

echo "Starting training..."
echo ""
"${TRAIN_CMD[@]}"

echo ""
echo -e "${GREEN}========================================"
echo "Training Completed Successfully!"
echo "========================================${NC}"
echo ""
echo "Artifacts:"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Log dir:    ${LOG_DIR}"
