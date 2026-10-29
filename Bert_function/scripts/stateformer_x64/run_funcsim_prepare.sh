#!/bin/bash

set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree

BASE_DIR="/path/to/numAnchor_artifact/stateformer_x64_dataset/raw_input/x86-64"
OUTPUT_DIR="/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2"
SYMBOLS_FILE="/path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/top_symbols_cfg_2inline.txt"
LOG_FILE="/path/to/numAnchor_artifact/log/Palmtree/funcsim_prepare_stateformer_x64_ctx2.log"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

python3 /path/to/numAnchor/PalmTree/data_generator/process_funcsim_batch.py \
  --base_dir "${BASE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --symbols_file "${SYMBOLS_FILE}" \
  --enable_smart_merge 2>&1 | tee -a "${LOG_FILE}"
