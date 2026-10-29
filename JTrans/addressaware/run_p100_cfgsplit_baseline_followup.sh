#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/conda/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

TRAIN_DIR="/path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100_cfgsplit_plt30"
TEST_DIR="/path/to/numAnchor_artifact/numAnchor/jtrans/official_small_test_cfgsplit_plt30"

PRETRAIN_OUT="${TRAIN_DIR}/pretrain_out_seq512_h384_l6_baseline"
VOCAB="${TRAIN_DIR}/vocab.pkl"

while [[ ! -f "${PRETRAIN_OUT}/history.json" ]]; do
  sleep 120
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/train_funcsim_paper.py" \
  --function_blocks "${TRAIN_DIR}/funcsim_function_blocks.json" \
  --funcsim_pairs "${TRAIN_DIR}/funcsim_pairs.json" \
  --vocab "${VOCAB}" \
  --pretrained_bert "${PRETRAIN_OUT}/best_model.pt" \
  --output_dir "${TRAIN_DIR}/paper_triplet_out_seq512_h384_l6_baseline_nofreeze_lr1e5" \
  --log_dir "${TRAIN_DIR}/logs_paper_triplet_seq512_h384_l6_baseline_nofreeze_lr1e5" \
  --hidden 384 \
  --n_layers 6 \
  --attn_heads 8 \
  --seq_len 512 \
  --epochs 10 \
  --lr 1e-5 \
  --device cuda \
  --gradient_checkpointing \
  --experiment_name "seq512_h384_l6_baseline_nofreeze_lr1e5" \
  --disable_address_embedding \
  --disable_var_embedding

while [[ ! -f "${TRAIN_DIR}/paper_triplet_out_seq512_h384_l6_baseline_nofreeze_lr1e5/best_model.pt" ]]; do
  sleep 120
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_opt_pairs_paper.py" \
  --function_blocks "${TEST_DIR}/funcsim_function_blocks.json" \
  --funcsim_pairs "${TEST_DIR}/funcsim_pairs.json" \
  --vocab "${VOCAB}" \
  --checkpoint "${TRAIN_DIR}/paper_triplet_out_seq512_h384_l6_baseline_nofreeze_lr1e5/best_model.pt" \
  --output "${TRAIN_DIR}/paper_triplet_out_seq512_h384_l6_baseline_nofreeze_lr1e5/official_small_test_opt_eval.json" \
  --batch_size 64 \
  --poolsizes "32,1000,10000" \
  --device cuda
