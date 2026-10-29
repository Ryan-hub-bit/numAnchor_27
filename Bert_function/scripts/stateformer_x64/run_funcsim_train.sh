#!/bin/bash

set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /path/to/numAnchor/PalmTree/funcsim

PRETRAINED_BERT="${PRETRAINED_BERT:-/path/to/numAnchor/output/hpc_pretrain/stateformer_x64_ctx2_conservative/best_bert.pt}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/numAnchor/output/palmtree_output/funcsim/stateformer_x64_ctx2}"
LOG_DIR="${LOG_DIR:-/path/to/numAnchor_artifact/log/Palmtree/funcsim_stateformer_x64_ctx2}"
DATASET_FRACTION="${DATASET_FRACTION:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NEGATIVE_SAMPLES="${NEGATIVE_SAMPLES:-0}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.9}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
DEVICE="${DEVICE:-cuda}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-funcsim_stateformer_x64_ctx2}"
TASK_NAME="${TASK_NAME:-stateformer_x64_ctx2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SEQ_LEN="${SEQ_LEN:-420}"
MAX_INSTRUCTIONS="${MAX_INSTRUCTIONS:-50}"
EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
WINDOW_SUBBATCH="${WINDOW_SUBBATCH:-256}"
FREEZE_BERT="${FREEZE_BERT:-false}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

EXTRA_ARGS=()
if [ -n "${RESUME_CHECKPOINT}" ]; then
  EXTRA_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi
if [ "${FREEZE_BERT}" = "true" ]; then
  EXTRA_ARGS+=(--freeze_bert)
fi
if [ "${GRADIENT_CHECKPOINTING}" = "true" ]; then
  EXTRA_ARGS+=(--gradient_checkpointing)
fi

python train.py \
  --function_blocks "/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json" \
  --funcsim_pairs "/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json" \
  --vocab "/path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/vocab/vocab_stateformer_x64_ctx2.pkl" \
  --pretrained_bert "${PRETRAINED_BERT}" \
  --hidden 768 \
  --n_layers 12 \
  --attn_heads 12 \
  --embedding_dim "${EMBEDDING_DIM}" \
  --seq_len "${SEQ_LEN}" \
  --max_instructions "${MAX_INSTRUCTIONS}" \
  --window_subbatch "${WINDOW_SUBBATCH}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-5}" \
  --lr "${LR:-3e-5}" \
  --lr_bert "${LR_BERT:-3e-6}" \
  --warmup_ratio "${WARMUP_RATIO:-0.06}" \
  --loss infonce \
  --temperature 0.07 \
  --margin 1.0 \
  --negative_samples "${NEGATIVE_SAMPLES}" \
  --dataset_fraction "${DATASET_FRACTION}" \
  --train_split "${TRAIN_SPLIT}" \
  --val_split "${VAL_SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --task_name "${TASK_NAME}" \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}"
