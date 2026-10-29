#!/bin/bash

set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree

CFG_ROOT="${CFG_ROOT:-/path/to/numAnchor_artifact/stateformer_x64_dataset/cfg_ctx2}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-/path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2}"
COMBINED="${COMBINED:-${PRETRAIN_ROOT}/cfg_2inline_pretrain.txt}"
TRAIN_OUT="${TRAIN_OUT:-${PRETRAIN_ROOT}/splits/train_cfg.txt}"
VAL_OUT="${VAL_OUT:-${PRETRAIN_ROOT}/splits/val_cfg.txt}"
LOG_FILE="${LOG_FILE:-/path/to/numAnchor_artifact/log/Palmtree/combine_and_split_ctx2.log}"
TOP_N="${TOP_N:-400}"
VAL_RATIO="${VAL_RATIO:-0.02}"
ENABLE_SMART_MERGE="${ENABLE_SMART_MERGE:-true}"
SAMPLE_RATIO="${SAMPLE_RATIO:-1.0}"

mkdir -p "${PRETRAIN_ROOT}" "$(dirname "${LOG_FILE}")"

{
  printf '[%s] Combining recursive ctx=2 CFG files\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  combine_args=(
    --input_dir "${CFG_ROOT}"
    --output_dir "${PRETRAIN_ROOT}"
    --output_name "$(basename "${COMBINED}")"
    --top_n "${TOP_N}"
    --sample_ratio "${SAMPLE_RATIO}"
    --recursive
  )
  if [ "${ENABLE_SMART_MERGE}" = "true" ]; then
    combine_args+=(--enable_smart_merge)
  fi
  python3 /path/to/numAnchor/PalmTree/data_generator/combine_cfg_2inline.py "${combine_args[@]}"

  printf '[%s] Splitting combined corpus into train/val\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 /path/to/numAnchor/PalmTree/scripts/stateformer_x64/split_pretrain_corpus.py \
    --input "${COMBINED}" \
    --train_output "${TRAIN_OUT}" \
    --val_output "${VAL_OUT}" \
    --val_ratio "${VAL_RATIO}" \
    --seed 42
} 2>&1 | tee -a "${LOG_FILE}"
