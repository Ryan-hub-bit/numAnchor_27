#!/usr/bin/env bash
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

cd /path/to/numAnchor/trex

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ROOT="${RUN_ROOT:-/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_baseline_finetune_20260422}"
CHECKPOINT="${1:-${RUN_ROOT}/baseline_similarity_ckpt/checkpoint_best.pt}"
DATA_DIR="${DATA_DIR:-${RUN_ROOT}/baseline_similarity_bin}"
FUNCTION_BLOCKS="${FUNCTION_BLOCKS:-/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json}"
FUNCSIM_PAIRS="${FUNCSIM_PAIRS:-/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/numAnchor/trex/output/stateformer_x64_baseline_similarity_20260422}"
MAX_POSITIONS="${MAX_POSITIONS:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p "${OUTPUT_DIR}"

python command/stateformer_x64/evaluate_pool.py \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --function-blocks "${FUNCTION_BLOCKS}" \
  --funcsim-pairs "${FUNCSIM_PAIRS}" \
  --pool-file /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_100.json \
  --mode baseline \
  --max-positions "${MAX_POSITIONS}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --output "${OUTPUT_DIR}/eval_pool_100.json"

python command/stateformer_x64/evaluate_pool.py \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --function-blocks "${FUNCTION_BLOCKS}" \
  --funcsim-pairs "${FUNCSIM_PAIRS}" \
  --pool-file /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_1k.json \
  --mode baseline \
  --max-positions "${MAX_POSITIONS}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --output "${OUTPUT_DIR}/eval_pool_1k.json"

python command/stateformer_x64/evaluate_pool.py \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --function-blocks "${FUNCTION_BLOCKS}" \
  --funcsim-pairs "${FUNCSIM_PAIRS}" \
  --pool-file /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_10k.json \
  --mode baseline \
  --max-positions "${MAX_POSITIONS}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --output "${OUTPUT_DIR}/eval_pool_10k.json"
