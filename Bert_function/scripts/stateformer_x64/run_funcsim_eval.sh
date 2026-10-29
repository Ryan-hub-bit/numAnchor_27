#!/bin/bash

set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

POOL_TAG="${1:-10k}"

cd /path/to/numAnchor/PalmTree/funcsim

python evaluate.py \
  --function_blocks "/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json" \
  --funcsim_pairs "/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json" \
  --vocab "/path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/vocab/vocab_stateformer_x64_ctx2.pkl" \
  --checkpoint "/path/to/numAnchor/output/palmtree_output/funcsim/stateformer_x64_ctx2/best_model.pt" \
  --hidden 768 \
  --n_layers 12 \
  --attn_heads 12 \
  --embedding_dim 256 \
  --seq_len 420 \
  --max_instructions 50 \
  --batch_size 64 \
  --task_name "stateformer_x64_ctx2_${POOL_TAG}" \
  --output "/path/to/numAnchor/output/palmtree_output/eval/stateformer_x64_ctx2/eval_results_${POOL_TAG}.json" \
  --log_dir "/path/to/numAnchor_artifact/log/Palmtree/eval_stateformer_x64_ctx2" \
  --device cuda \
  --eval_pool "/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_${POOL_TAG}.json"
