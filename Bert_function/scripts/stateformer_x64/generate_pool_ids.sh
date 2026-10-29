#!/bin/bash

set -euo pipefail

FUNCTION_BLOCKS="/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json"
FUNCSIM_PAIRS="/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json"
OUTPUT_DIR="/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2"
LOG_FILE="/path/to/numAnchor_artifact/log/Palmtree/generate_pool_ids_stateformer_x64_ctx2.log"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

for pool_size in 100 1000 10000; do
  python3 /path/to/numAnchor/PalmTree/funcsim/generate_pool_data.py \
    --function_blocks "${FUNCTION_BLOCKS}" \
    --funcsim_pairs "${FUNCSIM_PAIRS}" \
    --pool_size "${pool_size}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 2>&1 | tee -a "${LOG_FILE}"
done
