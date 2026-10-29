#!/usr/bin/env bash
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

cd /path/to/numAnchor/trex

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CHECKPOINT="${1:-/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/addressaware_aa_ckpt/checkpoint_best.pt}"
DATA_DIR="/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/baseline_bin"
FUNCTION_BLOCKS="/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json"
FUNCSIM_PAIRS="/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json"
OUTPUT_DIR="/path/to/numAnchor/trex/output/stateformer_x64_addressaware_aa_20260421"

mkdir -p "${OUTPUT_DIR}"

python command/stateformer_x64/evaluate_pool.py \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --function-blocks "${FUNCTION_BLOCKS}" \
  --funcsim-pairs "${FUNCSIM_PAIRS}" \
  --pool-file /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_100.json \
  --mode addressaware_aa \
  --max-positions 128 \
  --batch-size 64 \
  --device cuda \
  --output "${OUTPUT_DIR}/eval_pool_100.json"

python command/stateformer_x64/evaluate_pool.py \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --function-blocks "${FUNCTION_BLOCKS}" \
  --funcsim-pairs "${FUNCSIM_PAIRS}" \
  --pool-file /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_1k.json \
  --mode addressaware_aa \
  --max-positions 128 \
  --batch-size 64 \
  --device cuda \
  --output "${OUTPUT_DIR}/eval_pool_1k.json"

python command/stateformer_x64/evaluate_pool.py \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --function-blocks "${FUNCTION_BLOCKS}" \
  --funcsim-pairs "${FUNCSIM_PAIRS}" \
  --pool-file /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_10k.json \
  --mode addressaware_aa \
  --max-positions 128 \
  --batch-size 64 \
  --device cuda \
  --output "${OUTPUT_DIR}/eval_pool_10k.json"
