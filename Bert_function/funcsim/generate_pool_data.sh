#!/bin/bash
#
# Generate evaluation pool data files for pool sizes: 100, 1000, 10000
#
# Usage: ./generate_pool_data.sh

set -e

# Input
FUNCTION_BLOCKS="/path/to/numAnchor_artifact/addressaware/funcsim_block_8/function_blocks.json"
FUNCSIM_PAIRS="/path/to/numAnchor_artifact/addressaware/funcsim_block_8/funcsim_pairs.json"

# Output directory
OUTPUT_DIR="/path/to/numAnchor_artifact/addressaware/funcsim_block_8"

SEED=42

echo "========================================"
echo "Generating Evaluation Pool Data Files"
echo "========================================"
echo "Random seed:  ${SEED}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "========================================"
echo ""

for POOL_SIZE in 100 1000 10000; do
    echo "----------------------------------------"
    echo "Pool size: ${POOL_SIZE}"
    echo "----------------------------------------"
    python3 generate_pool_data.py \
      --function_blocks "${FUNCTION_BLOCKS}" \
      --funcsim_pairs   "${FUNCSIM_PAIRS}"   \
      --pool_size       ${POOL_SIZE}          \
      --output_dir      "${OUTPUT_DIR}"       \
      --seed            ${SEED}
    echo ""
done

echo "========================================"
echo "All pools generated:"
echo "  ${OUTPUT_DIR}/pool_ids_100.json"
echo "  ${OUTPUT_DIR}/pool_function_blocks_100.json"
echo "  ${OUTPUT_DIR}/pool_ids_1k.json"
echo "  ${OUTPUT_DIR}/pool_function_blocks_1k.json"
echo "  ${OUTPUT_DIR}/pool_ids_10k.json"
echo "  ${OUTPUT_DIR}/pool_function_blocks_10k.json"
echo "========================================"
